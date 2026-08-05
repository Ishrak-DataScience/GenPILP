# -*- coding: utf-8 -*-
"""
stage1b_large_scale_PLIP_mask_calculation.py
==============================================
Large-scale batch version of stage1_mask_calculation.py's PLIP masking.

stage1_mask_calculation.py processes a handful of hand-curated ligands
(config.PIPELINE_INPUTS: 5 entries, each a manually specified
{pdb_path, plip_xml_path, resname, chain, resseq}). This script does the
same masking (same functions, unmodified: parse_plip_xml_v2_select,
run_pipeline) but scales it up: it selects PDB IDs (either a random sample
of N, or every available id) from a large local PDB mirror + a matching
pre-computed PLIP-XML directory, and masks every binding site PLIP found
in each selected structure.

    Local PDB mirror   :  <pdb_root>/<mid2>/pdb<id>.ent.gz
                           (mid2 = id[1:3], e.g. "100d" -> "00")
    Pre-computed PLIP   :  <xml_root>/pdb<id>.xml   (flat)

Only PDB IDs present in BOTH locations are eligible for sampling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN DECISIONS (resolved with the user; see conversation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
D1. Binding-site selection: NO filtering/heuristics are added beyond what
    stage1_mask_calculation.py already does. A PLIP XML can list several
    <bindingsite> blocks per structure (e.g. pdb100d.xml has 3: two
    nucleotide copies + one spermine) — every one of them is masked, same
    as stage1_mask_calculation.py would if each were listed separately in
    config.PIPELINE_INPUTS. No blocklist, no "largest ligand" heuristic.

D2. Sample size N = number of PDB/XML pairs sampled (not number of
    resulting masked rows) — each sampled structure contributes as many
    output rows as it has binding sites. Sampling mode "all" processes
    every eligible PDB/XML pair instead of a random N of them (N/seed are
    ignored); "eligible" already means present in BOTH the PDB mirror and
    the PLIP XML directory (see discover_available_ids) — ids missing
    either file are skipped automatically, same as in "n" mode.

D3. Masking mode: identical prompt/default to stage1_mask_calculation.py's
    main() (mode 1 = INTERACTION masking by default, mode 2 =
    NON-INTERACTION). Overridable non-interactively via --mode.

D4. Per-ligand outputs: identical to stage1_mask_calculation.py —
    run_pipeline() is called unmodified with out_prefix set, so it still
    writes one .meta.json + one 2D interaction PNG per masked binding
    site. On top of that (necessary at this scale, not present in the
    original since it only ever handled 5 ligands printed to console),
    this script also writes ONE aggregated summary CSV across every
    sampled (pdb_id, resname, chain, resseq) row, successes and failures.

D5. CLI args mirror every interactive prompt (--sampling-mode, --n, --seed,
    --pdb-root, --xml-root, --output-dir, --mode, --include-types).
    Any argument supplied on the command line skips its interactive
    prompt — so the same script runs either as a batch job (all args
    passed, zero prompts) or standalone (prompts for whatever wasn't
    passed), matching every other stage script's convention.

D6. stage1_mask_calculation.py itself is UNCHANGED — this script only
    imports and reuses its functions.

D7. Resume/checkpoint (needed once N can mean "all 220,000+ structures" —
    a run that long will eventually get interrupted): summary CSV rows are
    flushed + fsync'd to disk after every PDB ID, not buffered until the
    end. By default (resume=True / no --no-resume), if the summary CSV in
    --output-dir already exists, PDB ids already present in it are skipped
    on restart, so a killed/crashed run can just be re-launched with the
    same args and it picks up where it left off. --no-resume forces a
    clean overwrite instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Standalone (interactive prompts for anything not passed):
        python stage1b_large_scale_PLIP_mask_calculation.py

    Batch, fixed-size random sample (no prompts — every value supplied):
        python stage1b_large_scale_PLIP_mask_calculation.py \\
            --sampling-mode n --n 500 --seed 42 --mode 2 \\
            --pdb-root /group/bioinf_tmp/Data/pdb \\
            --xml-root /group/bioinf_tmp/plip_pdb2xml \\
            --output-dir /path/to/output

    Batch, ALL available PDB/XML pairs (skips ids missing either file;
    safe to re-run after a crash/interrupt — already-processed ids in
    --output-dir's summary CSV are skipped automatically, use --no-resume
    to force a clean restart instead):
        python stage1b_large_scale_PLIP_mask_calculation.py \\
            --sampling-mode all --mode 2 \\
            --pdb-root /group/bioinf_tmp/Data/pdb \\
            --xml-root /group/bioinf_tmp/plip_pdb2xml \\
            --output-dir /path/to/output

HOW TO TEST (uses the pdb100d fixture already in Dataset/, no network mount needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage1b_large_scale_PLIP_mask_calculation.py --test
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import random
import re
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import config
from stage1_mask_calculation import run_pipeline

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else []


CSV_FIELDS = [
    "pdb_id", "resname", "chain", "resseq", "masking_mode",
    "status", "error",
    "smiles", "masked_smiles", "bpe_mask_count", "masked_atom_indices",
]

_XML_ID_RE = re.compile(r"^pdb(.+)\.xml$", re.IGNORECASE)


# ════════════════════════════════════════════════════════════════════════════
#  PDB-ID DISCOVERY + SAMPLING
# ════════════════════════════════════════════════════════════════════════════

def _pdb_gz_path(pdb_root: str, pdb_id: str) -> str:
    """<pdb_root>/<mid2>/pdb<id>.ent.gz, mid2 = id[1:3] (e.g. '100d' -> '00')."""
    mid2 = pdb_id[1:3] if len(pdb_id) >= 3 else pdb_id
    return os.path.join(pdb_root, mid2, f"pdb{pdb_id}.ent.gz")


def discover_available_ids(pdb_root: str, xml_root: str) -> List[str]:
    """
    PDB IDs eligible for sampling: present as <xml_root>/pdb<id>.xml AND
    with a matching gzipped structure at <pdb_root>/<mid2>/pdb<id>.ent.gz.
    """
    ids: List[str] = []
    for xml_path in glob.glob(os.path.join(xml_root, "pdb*.xml")):
        m = _XML_ID_RE.match(os.path.basename(xml_path))
        if not m:
            continue
        pdb_id = m.group(1).lower()
        if os.path.isfile(_pdb_gz_path(pdb_root, pdb_id)):
            ids.append(pdb_id)
    return sorted(ids)


def sample_ids(available: List[str], n: int, seed: int) -> List[str]:
    if n >= len(available):
        if n > len(available):
            print(
                f"  ⚠️  Requested n={n} but only {len(available)} PDB/XML pairs "
                f"are available — using all {len(available)}."
            )
        return list(available)
    rng = random.Random(seed)
    return sorted(rng.sample(available, n))


def _load_completed_ids(summary_path: str) -> set:
    """PDB IDs already present in an existing summary CSV (used to resume a run)."""
    if not os.path.isfile(summary_path):
        return set()
    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["pdb_id"] for row in reader if row.get("pdb_id")}


# ════════════════════════════════════════════════════════════════════════════
#  BINDING-SITE DISCOVERY (D1: enumerate, do not filter)
# ════════════════════════════════════════════════════════════════════════════

def list_binding_sites(xml_path: str) -> List[Tuple[str, Optional[str], Optional[int]]]:
    """
    Return every (hetid, chain, position) identifier triple found across all
    <bindingsite> blocks in a PLIP XML file — no filtering (D1). Each triple
    is later fed as (resname, chain, resseq) into the unmodified
    stage1_mask_calculation.parse_plip_xml_v2_select / run_pipeline.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    sites: List[Tuple[str, Optional[str], Optional[int]]] = []
    for bs in root.findall(".//bindingsite"):
        ids = bs.find("./identifiers")
        if ids is None:
            continue
        hetid = (ids.findtext("hetid") or "").strip()
        if not hetid:
            continue
        chain = (ids.findtext("chain") or "").strip() or None
        pos_text = (ids.findtext("position") or "").strip()
        try:
            pos = int(pos_text) if pos_text else None
        except ValueError:
            pos = None
        sites.append((hetid, chain, pos))
    return sites


# ════════════════════════════════════════════════════════════════════════════
#  PDB DECOMPRESSION
# ════════════════════════════════════════════════════════════════════════════

def decompress_pdb_gz(gz_path: str, dest_dir: str) -> str:
    """Gunzip <gz_path> into dest_dir, return the path to the plain-text .ent file."""
    out_path = os.path.join(dest_dir, os.path.basename(gz_path)[:-3])  # strip ".gz"
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


# ════════════════════════════════════════════════════════════════════════════
#  PER-STRUCTURE PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def _row_from_meta(pdb_id: str, resname: str, chain: Optional[str],
                    resseq: Optional[int], meta: dict) -> dict:
    return {
        "pdb_id":              pdb_id,
        "resname":             resname,
        "chain":               chain,
        "resseq":              resseq,
        "masking_mode":        meta["masking_mode"],
        "status":              "ok",
        "error":               "",
        "smiles":              meta["smiles"],
        "masked_smiles":       meta.get("masked_smiles", ""),
        "bpe_mask_count":      meta.get("bpe_mask_count", ""),
        "masked_atom_indices": json.dumps(meta["masked_atom_indices"]),
    }


def _error_row(pdb_id: str, resname: str, chain: Optional[str],
                resseq: Optional[int], mask_non_attractive: bool, err: Exception) -> dict:
    return {
        "pdb_id":              pdb_id,
        "resname":             resname,
        "chain":               chain,
        "resseq":              resseq,
        "masking_mode":        "non-attractive" if mask_non_attractive else "attractive",
        "status":              "error",
        "error":               str(err),
        "smiles":              "",
        "masked_smiles":       "",
        "bpe_mask_count":      "",
        "masked_atom_indices": "",
    }


def process_pdb_id(
    pdb_id: str,
    pdb_root: str,
    xml_root: str,
    output_dir: str,
    include_types: List[str],
    mask_non_attractive: bool,
) -> List[dict]:
    """Decompress + mask every binding site found for one PDB ID. Returns CSV rows."""
    xml_path = os.path.join(xml_root, f"pdb{pdb_id}.xml")
    gz_path = _pdb_gz_path(pdb_root, pdb_id)

    try:
        sites = list_binding_sites(xml_path)
    except Exception as e:
        return [_error_row(pdb_id, "", None, None, mask_non_attractive, e)]

    if not sites:
        return [_error_row(
            pdb_id, "", None, None, mask_non_attractive,
            ValueError("No <bindingsite> entries in PLIP XML"),
        )]

    rows: List[dict] = []
    with tempfile.TemporaryDirectory() as td:
        try:
            pdb_path = decompress_pdb_gz(gz_path, td)
        except Exception as e:
            return [_error_row(pdb_id, "", None, None, mask_non_attractive, e)]

        for resname, chain, resseq in sites:
            tag = f"{pdb_id}_{resname}_{chain}_{resseq}"
            try:
                meta = run_pipeline(
                    pdb_path            = pdb_path,
                    plip_xml_path       = xml_path,
                    resname             = resname,
                    chain               = chain,
                    resseq              = resseq,
                    include_types       = include_types,
                    representation      = "selfies",
                    mask_token          = "<mask>",
                    out_prefix          = os.path.join(output_dir, tag + "_masked.selfies"),
                    serial_map_json     = None,
                    mask_non_attractive = mask_non_attractive,
                )
                rows.append(_row_from_meta(pdb_id, resname, chain, resseq, meta))
            except Exception as e:
                rows.append(_error_row(pdb_id, resname, chain, resseq, mask_non_attractive, e))

    return rows


# ════════════════════════════════════════════════════════════════════════════
#  BATCH DRIVER
# ════════════════════════════════════════════════════════════════════════════

def run_large_scale_plip_masking(
    n: int,
    seed: int,
    pdb_root: str,
    xml_root: str,
    output_dir: str,
    mask_non_attractive: bool,
    include_types: Optional[List[str]] = None,
    sampling_mode: str = "n",
    resume: bool = True,
) -> str:
    """
    Select PDB IDs (either a random sample of size n, or — sampling_mode="all" —
    every available PDB/XML pair), mask every binding site PLIP found in each,
    and write one aggregated summary CSV. Returns the summary CSV path.

    Rows are flushed to the summary CSV after each PDB ID (not buffered in
    memory until the end) so a run over the full 220k+ dataset can be safely
    interrupted. If resume=True and the summary CSV already exists, PDB IDs
    already recorded in it are skipped (D2: skip already-processed ids, not
    already-processed rows — process_pdb_id() writes all rows for an id in
    one call, so "present in the CSV" implies "fully processed").
    """
    include_types = include_types or list(config.INCLUDE_TYPES)
    os.makedirs(output_dir, exist_ok=True)

    available = discover_available_ids(pdb_root, xml_root)
    if not available:
        raise RuntimeError(
            f"No PDB/XML pairs found (pdb_root={pdb_root}, xml_root={xml_root})."
        )

    if sampling_mode == "all":
        picked = list(available)
        print(f"  ALL mode: {len(available)} PDB/XML pairs available (PDB or PLIP-missing ids already excluded).")
    else:
        picked = sample_ids(available, n, seed)
        print(f"  {len(available)} PDB/XML pairs available; sampled {len(picked)} (seed={seed}).")

    summary_path = os.path.join(output_dir, "stage1b_large_scale_plip_mask_summary.csv")

    completed: set = set()
    if resume:
        completed = _load_completed_ids(summary_path)
        if completed:
            before = len(picked)
            picked = [pid for pid in picked if pid not in completed]
            print(f"  Resume: {before - len(picked)} of {before} already in {summary_path} — skipping them.")
    file_exists_with_header = resume and os.path.isfile(summary_path) and bool(completed)
    file_mode = "a" if file_exists_with_header else "w"

    with open(summary_path, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if file_mode == "w":
            writer.writeheader()
        for pdb_id in tqdm(picked, desc="Masking PDB structures", unit="pdb"):
            rows = process_pdb_id(
                pdb_id, pdb_root, xml_root, output_dir, include_types, mask_non_attractive,
            )
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

    with open(summary_path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    n_ok = sum(1 for r in all_rows if r["status"] == "ok")
    print(f"  {n_ok}/{len(all_rows)} binding sites masked successfully (cumulative across resumes).")
    print(f"  Summary CSV: {summary_path}")
    return summary_path


# ════════════════════════════════════════════════════════════════════════════
#  CLI / INTERACTIVE PROMPTS  (D5)
# ════════════════════════════════════════════════════════════════════════════

def _prompt_str(question: str, default: str) -> str:
    raw = input(f"  {question} [{default}]: ").strip()
    return raw or default


def _prompt_int(question: str, default: int) -> int:
    while True:
        raw = input(f"  {question} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("    Please enter an integer.")


def _ask_mode() -> bool:
    """Identical prompt/default to stage1_mask_calculation.py's main() (D3)."""
    print("""
  Masking mode:
    1 : INTERACTION      (mask_non_attractive = False)  [default]
        Atoms that DO participate in protein-ligand interactions are masked.
    2 : NON-INTERACTION  (mask_non_attractive = True)
        Atoms that do NOT participate in interactions are masked.
""")
    while True:
        raw = input("  Select masking mode (1 / 2) [1]: ").strip()
        if raw in ("", "1"):
            return False
        if raw == "2":
            return True
        print("    Please type 1 or 2.")


def _ask_sampling_mode() -> str:
    print("""
  Sampling mode:
    1 : Fixed sample size N, drawn at random             [default]
    2 : ALL available PDB/XML pairs (skip ids missing a PDB or PLIP XML file)
""")
    while True:
        raw = input("  Select sampling mode (1 / 2) [1]: ").strip()
        if raw in ("", "1"):
            return "n"
        if raw == "2":
            return "all"
        print("    Please type 1 or 2.")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Large-scale PLIP mask calculation over a random sample "
                     "(or all) of PDB structures."
    )
    p.add_argument("--sampling-mode", type=str, choices=["n", "all"], default=None,
                    help="'n' = sample --n PDB/XML pairs at random (default); "
                         "'all' = process every available pair, skipping ids "
                         "missing a PDB or PLIP XML file.")
    p.add_argument("--n", type=int, default=None, help="Number of PDB/XML pairs to sample (--sampling-mode n).")
    p.add_argument("--seed", type=int, default=None, help="Random sample seed (--sampling-mode n).")
    p.add_argument("--pdb-root", type=str, default=None,
                    help="Root of the local PDB mirror (<root>/<mid2>/pdb<id>.ent.gz).")
    p.add_argument("--xml-root", type=str, default=None,
                    help="Directory of pre-computed PLIP XML files (<root>/pdb<id>.xml).")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    p.add_argument("--mode", type=str, choices=["1", "2"], default=None,
                    help="1 = INTERACTION masking, 2 = NON-INTERACTION masking.")
    p.add_argument("--include-types", type=str, default=None,
                    help="Comma-separated PLIP interaction types (default: config.INCLUDE_TYPES).")
    p.add_argument("--no-resume", action="store_true",
                    help="Disable resume/checkpoint behavior — by default, PDB ids already "
                         "recorded in an existing summary CSV in --output-dir are skipped, "
                         "so an interrupted run can continue where it left off.")
    p.add_argument("--test", action="store_true", help="Run the self-test and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    print("\n" + "=" * 60)
    print("STAGE 1b: LARGE-SCALE PLIP MASK CALCULATION")
    print("=" * 60)

    args = _parse_args(argv)

    sampling_mode = args.sampling_mode or _ask_sampling_mode()
    if sampling_mode == "n":
        n = args.n if args.n is not None else _prompt_int(
            "Sample size N", getattr(config, "STAGE1B_PLIP_SAMPLE_N", 500)
        )
        seed = args.seed if args.seed is not None else _prompt_int(
            "Random sample seed", getattr(config, "STAGE1B_PLIP_SAMPLE_SEED", 42)
        )
    else:
        n = args.n if args.n is not None else 0
        seed = args.seed if args.seed is not None else getattr(config, "STAGE1B_PLIP_SAMPLE_SEED", 42)
    resume = getattr(config, "STAGE1B_RESUME", True) and not args.no_resume
    pdb_root = args.pdb_root or _prompt_str(
        "PDB mirror root", config.PLIP_LARGE_SCALE_PDB_ROOT
    )
    xml_root = args.xml_root or _prompt_str(
        "PLIP XML root", config.PLIP_LARGE_SCALE_XML_ROOT
    )
    output_dir = args.output_dir or _prompt_str(
        "Output directory", config.STAGE1B_PLIP_MASK_DIR
    )
    mask_non_attractive = (
        args.mode == "2" if args.mode is not None else _ask_mode()
    )
    include_types = (
        [t.strip() for t in args.include_types.split(",") if t.strip()]
        if args.include_types else list(config.INCLUDE_TYPES)
    )

    print(f"""
  Sampling mode    : {"ALL available PDB/XML pairs" if sampling_mode == "all" else f"N={n} (seed={seed})"}
  PDB root         : {pdb_root}
  XML root         : {xml_root}
  Output dir       : {output_dir}
  Masking mode     : {"NON-INTERACTION" if mask_non_attractive else "INTERACTION"}
  Include types    : {include_types}
  Resume           : {resume}
""")

    run_large_scale_plip_masking(
        n=n, seed=seed, pdb_root=pdb_root, xml_root=xml_root,
        output_dir=output_dir, mask_non_attractive=mask_non_attractive,
        include_types=include_types, sampling_mode=sampling_mode, resume=resume,
    )

    print("\n✅ Stage 1b large-scale PLIP masking complete.")


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST (uses the pdb100d fixture already checked into Dataset/)
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_xml_root = os.path.join(repo_dir, "Dataset")
    fixture_pdb_root = os.path.join(repo_dir, "Dataset")  # contains 00/pdb100d.ent.gz

    xml_path = os.path.join(fixture_xml_root, "pdb100d.xml")
    gz_path = _pdb_gz_path(fixture_pdb_root, "100d")
    assert os.path.isfile(xml_path), f"Missing test fixture: {xml_path}"
    assert os.path.isfile(gz_path), f"Missing test fixture: {gz_path}"

    available = discover_available_ids(fixture_pdb_root, fixture_xml_root)
    assert "100d" in available, f"Expected '100d' in discovered ids, got {available}"

    picked = sample_ids(available, n=1, seed=0)
    assert picked == ["100d"]

    sites = list_binding_sites(xml_path)
    assert len(sites) == 3, f"Expected 3 binding sites in pdb100d.xml, got {len(sites)}"
    hetids = {s[0] for s in sites}
    assert hetids == {"C", "SPM"}, f"Unexpected hetids: {hetids}"

    with tempfile.TemporaryDirectory() as td:
        summary_path = run_large_scale_plip_masking(
            n=1, seed=0,
            pdb_root=fixture_pdb_root, xml_root=fixture_xml_root,
            output_dir=td, mask_non_attractive=True,
        )
        assert os.path.isfile(summary_path)
        with open(summary_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3, f"Expected 3 summary rows (one per binding site), got {len(rows)}"
        for row in rows:
            assert row["pdb_id"] == "100d"
            assert row["status"] in ("ok", "error")

        # Resume (default): rerunning against the same output_dir must skip
        # "100d" (already in the summary CSV) and leave it untouched (D7).
        summary_path_resumed = run_large_scale_plip_masking(
            n=1, seed=0,
            pdb_root=fixture_pdb_root, xml_root=fixture_xml_root,
            output_dir=td, mask_non_attractive=True, resume=True,
        )
        with open(summary_path_resumed, newline="", encoding="utf-8") as f:
            rows_resumed = list(csv.DictReader(f))
        assert rows_resumed == rows, "Resume should leave already-completed rows untouched"

        # --no-resume equivalent (resume=False): must overwrite from scratch.
        summary_path_fresh = run_large_scale_plip_masking(
            n=1, seed=0,
            pdb_root=fixture_pdb_root, xml_root=fixture_xml_root,
            output_dir=td, mask_non_attractive=True, resume=False,
        )
        with open(summary_path_fresh, newline="", encoding="utf-8") as f:
            rows_fresh = list(csv.DictReader(f))
        assert len(rows_fresh) == 3, f"Expected 3 fresh rows after --no-resume, got {len(rows_fresh)}"

    with tempfile.TemporaryDirectory() as td:
        # sampling_mode="all": with only one eligible pair in the fixture dirs,
        # this must behave the same as n=1 (D2).
        summary_path_all = run_large_scale_plip_masking(
            n=0, seed=0,
            pdb_root=fixture_pdb_root, xml_root=fixture_xml_root,
            output_dir=td, mask_non_attractive=True, sampling_mode="all",
        )
        with open(summary_path_all, newline="", encoding="utf-8") as f:
            rows_all = list(csv.DictReader(f))
        assert len(rows_all) == 3, f"Expected 3 rows in ALL mode, got {len(rows_all)}"
        assert all(row["pdb_id"] == "100d" for row in rows_all)

    print("✅ Stage 1b large-scale PLIP masking self-test passed.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main()

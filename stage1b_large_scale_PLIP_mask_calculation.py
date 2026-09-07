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

D4. Per-ligand outputs: the masking pass writes NO per-ligand artifacts —
    run_pipeline() is called with save_meta=False, save_plot=False. At
    500k+ binding sites, one .meta.json + one PNG each is millions of
    inodes for files nothing reads: every downstream consumer of Stage 1b
    (Stage 9's collect_pairs_from_stage1b, Stage 9a, and
    stage1a_random_masking_from_stage1b) reads ONLY the aggregated summary
    CSV this script writes, never the .meta.json files. Verification
    figures come from the separate sampled plot pass instead (D9).

D5. CLI args mirror every interactive prompt (--sampling-mode, --n, --seed,
    --pdb-root, --xml-root, --output-dir, --mode, --include-types).
    Any argument supplied on the command line skips its interactive
    prompt — so the same script runs either as a batch job (all args
    passed, zero prompts) or standalone (prompts for whatever wasn't
    passed), matching every other stage script's convention.

D6. stage1_mask_calculation.py is reused, not reimplemented: the masking
    itself (parse_plip_xml_v2_select, run_pipeline) is still its code,
    called unmodified. It was amended in exactly one respect, to serve D4
    and D9: run_pipeline() gained save_meta / save_plot / plot_path
    keyword arguments so a caller can choose per invocation whether the
    .meta.json and the figure get written. Their defaults reproduce the
    previous behavior byte for byte (meta whenever out_prefix is set, plot
    iff config.MASK_CALC_SAVE_PLOTS is truthy), so Stage 1's own main() is
    untouched; only this script passes them.

D7. Resume/checkpoint (needed once N can mean "all 220,000+ structures" —
    a run that long will eventually get interrupted): summary CSV rows are
    flushed + fsync'd to disk after every PDB ID, not buffered until the
    end. By default (resume=True / no --no-resume), if the summary CSV in
    --output-dir already exists, PDB ids already present in it are skipped
    on restart, so a killed/crashed run can just be re-launched with the
    same args and it picks up where it left off. --no-resume forces a
    clean overwrite instead.

    The CSV to resume from defaults to the one in --output-dir, but
    --resume-from / config.STAGE1B_RESUME_FROM_CSV points the run at an
    arbitrary CSV instead: its pdb_ids are skipped and new rows are
    APPENDED to that same file, so a summary carried over from an earlier
    run (or another machine) can be topped up in place. Its header must
    match CSV_FIELDS exactly or the run aborts rather than silently
    appending misaligned columns.

    An id is "done" if it appears in the CSV at all, including as an error
    row — a structure that failed is not retried on the next run. Pass
    --retry-errors / config.STAGE1B_RETRY_ERRORS to reprocess them: every
    id owning at least one status=="error" row is dropped from the CSV
    (rewritten atomically via a temp file + os.replace, once, at startup)
    and re-queued, so retries replace the old rows instead of duplicating
    the id's successful ones.

D8. Multiprocessing: process_pdb_id() (decompress + parse + mask one PDB
    ID) is CPU-bound and independent across PDB IDs, so it's dispatched
    to a concurrent.futures.ProcessPoolExecutor — one OS process per
    worker (sidesteps RDKit/matplotlib not being thread-parallel, and
    isolates a hard crash on one structure to one worker). The MAIN
    process remains the sole writer of the summary CSV (rows are written
    + flushed as each worker's future completes via as_completed, in
    completion order, not submission order) so there's no multi-writer
    contention and the resume/checkpoint guarantee from D7 still holds.
    --workers controls pool size; default is auto-detected from
    $SLURM_CPUS_PER_TASK (falls back to os.sched_getaffinity, then
    os.cpu_count()) — NOT bare os.cpu_count(), which reports the node's
    total CPUs and would over-subscribe under a SLURM --cpus-per-task
    allocation smaller than the node.

D9. Verification figures: drawing a 2D interaction plot for every one of
    ~500k binding sites is the single most expensive side effect in the
    script (a 12x12in/300dpi matplotlib render each) and exists only to
    eyeball that the masking is right. So config.MASK_CALC_SAVE_PLOTS is
    read here as TRI-STATE rather than a bare on/off:

        False / 0   no figures at all
        True        a figure for every masked binding site (old behavior)
        int N > 0   figures for exactly N randomly chosen binding sites

    N is drawn AFTER the masking pass, by sampling N rows uniformly from
    the finished summary CSV's status=="ok" rows (seeded by
    config.STAGE1B_PLOT_SEED) and re-running run_pipeline on just those
    with save_plot=True. Sampling from the completed CSV rather than
    gating inside the hot loop is what makes the count exactly N: the
    number of binding sites in the corpus is not known until the masking
    is done, since it takes parsing every XML to count them. It also means
    the plot pass is replayable on its own against any existing summary
    CSV (--plot-only, D10) with a different seed, to spot-check a fresh
    sample without redoing any masking. Sampled ligands keep their
    .meta.json (masked_atoms_detail is what you check the figure against);
    the other ~500k still do not.

D10. --plot-only skips the masking pass entirely and runs just D9's
    sampled plot pass over an existing summary CSV (--resume-from's, or
    --output-dir's). Nothing is masked and the CSV is not written.

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
    to force a clean restart instead), using every core SLURM gave this
    task (--workers omitted => auto-detected from $SLURM_CPUS_PER_TASK):
        python stage1b_large_scale_PLIP_mask_calculation.py \\
            --sampling-mode all --mode 2 \\
            --pdb-root /group/bioinf_tmp/Data/pdb \\
            --xml-root /group/bioinf_tmp/plip_pdb2xml \\
            --output-dir /path/to/output

    Same, but capped at 8 worker processes instead of auto-detected:
        python stage1b_large_scale_PLIP_mask_calculation.py \\
            --sampling-mode all --mode 2 --workers 8 \\
            --pdb-root /group/bioinf_tmp/Data/pdb \\
            --xml-root /group/bioinf_tmp/plip_pdb2xml \\
            --output-dir /path/to/output

HOW TO TEST (uses the pdb100d fixture already in Dataset/, no network mount needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage1b_large_scale_PLIP_mask_calculation.py --test

    Top up a summary CSV carried over from an earlier run, retrying the
    structures that errored in it:
        python stage1b_large_scale_PLIP_mask_calculation.py \\
            --sampling-mode all --mode 2 --retry-errors \\
            --resume-from /old/run/stage1b_large_scale_plip_mask_summary.csv \\
            --pdb-root /group/bioinf_tmp/Data/pdb \\
            --xml-root /group/bioinf_tmp/plip_pdb2xml \\
            --output-dir /path/to/output

    Draw 200 verification figures from an already-finished summary CSV,
    without masking anything (D10):
        python stage1b_large_scale_PLIP_mask_calculation.py \\
            --plot-only --plot-sample-n 200 \\
            --pdb-root /group/bioinf_tmp/Data/pdb \\
            --xml-root /group/bioinf_tmp/plip_pdb2xml \\
            --output-dir /path/to/output
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import config
from stage1_mask_calculation import run_pipeline

try:
    from tqdm import tqdm
except ImportError:
    # A silent stand-in with the same call surface, tqdm.write included;
    # the old inline stub was a bare function and had no .write.
    from tqdm_compat import tqdm  # type: ignore[misc]


CSV_FIELDS = [
    "pdb_id", "resname", "chain", "resseq", "masking_mode",
    "status", "error",
    "smiles", "masked_smiles", "bpe_mask_count", "masked_atom_indices",
]

_XML_ID_RE = re.compile(r"^pdb(.+)\.xml$", re.IGNORECASE)


def _default_worker_count() -> int:
    """
    Worker-process count to use when --workers isn't given (D8). Deliberately
    NOT bare os.cpu_count(): under SLURM, that reports the whole node's CPU
    count, not this job's --cpus-per-task allocation, and would over-subscribe.
    Preference order: $SLURM_CPUS_PER_TASK -> sched_getaffinity (POSIX,
    reflects cgroup/affinity limits) -> os.cpu_count() (e.g. Windows dev boxes).
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return max(1, len(sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


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


SUMMARY_CSV_NAME = "stage1b_large_scale_plip_mask_summary.csv"


def _check_summary_header(summary_path: str) -> bool:
    """
    True if summary_path exists and already carries a usable header row.

    Raises if it exists with a header that isn't CSV_FIELDS: appending to such
    a file would silently interleave rows under the wrong columns, and at this
    scale that corruption would not be noticed until Stage 9 read it back.
    """
    if not os.path.isfile(summary_path):
        return False
    with open(summary_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header is None:
        return False           # exists but empty — treat as fresh, rewrite header
    if header != CSV_FIELDS:
        raise RuntimeError(
            f"Refusing to append to {summary_path}: its header does not match this "
            f"scripts columns.\n    found:    {header}"
            f"\n    expected: {CSV_FIELDS}"
        )
    return True


def _load_completed_ids(summary_path: str) -> set:
    """PDB IDs already present in an existing summary CSV (used to resume a run)."""
    if not os.path.isfile(summary_path):
        return set()
    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["pdb_id"] for row in reader if row.get("pdb_id")}


def _drop_error_ids(summary_path: str) -> Tuple[set, int]:
    """
    --retry-errors support: remove every row belonging to a pdb_id that has at
    least one status=="error" row, so those ids are re-queued and their retry
    REPLACES the old rows instead of duplicating the id's successful ones.

    Returns (still-completed ids, number of rows dropped). Rewrites the CSV
    through a temp file in the same directory + os.replace, so an interruption
    mid-rewrite leaves the original summary intact rather than a half-file.
    """
    if not os.path.isfile(summary_path):
        return set(), 0

    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    bad = {r["pdb_id"] for r in rows if r.get("pdb_id") and r.get("status") == "error"}
    if not bad:
        return {r["pdb_id"] for r in rows if r.get("pdb_id")}, 0

    keep = [r for r in rows if r.get("pdb_id") not in bad]
    d = os.path.dirname(os.path.abspath(summary_path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".stage1b_summary_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(keep)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, summary_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return {r["pdb_id"] for r in keep if r.get("pdb_id")}, len(rows) - len(keep)


def resolve_plot_sample_n() -> int:
    """
    config.MASK_CALC_SAVE_PLOTS read as Stage 1b's tri-state plot control (D9).

        False / 0  ->  0   no interaction figures at all
        True       -> -1   a figure for every masked binding site
        int N > 0  ->  N   figures for exactly N randomly sampled binding sites

    bool is a subclass of int, so True/False are matched by identity first —
    int(True) would otherwise read as "exactly 1 figure".
    """
    v = getattr(config, "MASK_CALC_SAVE_PLOTS", True)
    if v is True:
        return -1
    if v is False or v is None:
        return 0
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return -1


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
                    # D4: the bulk pass writes neither artifact. out_prefix is
                    # still passed so the plot pass can reproduce the same
                    # naming, but with both flags off run_pipeline touches disk
                    # only via the summary CSV the main process writes.
                    save_meta           = False,
                    save_plot           = False,
                )
                rows.append(_row_from_meta(pdb_id, resname, chain, resseq, meta))
            except Exception as e:
                rows.append(_error_row(pdb_id, resname, chain, resseq, mask_non_attractive, e))

    return rows


# ════════════════════════════════════════════════════════════════════════════
#  SAMPLED VERIFICATION PLOTS  (D9)
# ════════════════════════════════════════════════════════════════════════════

def _plot_filename(pdb_id: str, resname: str, chain: Optional[str],
                    resseq: Optional[int], ext: str) -> str:
    """<pdb_id>.2d_interactions_<resname>_<chain>_<resseq>.<ext>"""
    return f"{pdb_id}.2d_interactions_{resname}_{chain}_{resseq}.{ext}"


def plot_pdb_id(
    pdb_id: str,
    sites: List[Tuple[str, Optional[str], Optional[int]]],
    pdb_root: str,
    xml_root: str,
    output_dir: str,
    plot_dir: str,
    include_types: List[str],
    mask_non_attractive: bool,
) -> Tuple[int, List[str]]:
    """
    Re-run run_pipeline for the sampled binding sites of ONE pdb_id with the
    figure (and .meta.json) turned back on. Grouped by pdb_id so a structure
    contributing several sampled sites is decompressed once, not once per site.

    Returns (n_drawn, error strings).
    """
    xml_path = os.path.join(xml_root, f"pdb{pdb_id}.xml")
    gz_path = _pdb_gz_path(pdb_root, pdb_id)
    ext = getattr(config, "MASK_CALC_PLOT_FORMAT", "png").lower().lstrip(".")

    drawn, errors = 0, []
    with tempfile.TemporaryDirectory() as td:
        try:
            pdb_path = decompress_pdb_gz(gz_path, td)
        except Exception as e:
            return 0, [f"{pdb_id}: {e}"]

        for resname, chain, resseq in sites:
            # out_prefix carries only the pdb_id: run_pipeline appends
            # _{resname}_{chain}_{resseq} to the .meta.json name itself, so
            # including the full tag here would double it in the filename.
            try:
                run_pipeline(
                    pdb_path            = pdb_path,
                    plip_xml_path       = xml_path,
                    resname             = resname,
                    chain               = chain,
                    resseq              = resseq,
                    include_types       = include_types,
                    representation      = "selfies",
                    mask_token          = "<mask>",
                    out_prefix          = os.path.join(output_dir, pdb_id + "_masked.selfies"),
                    serial_map_json     = None,
                    mask_non_attractive = mask_non_attractive,
                    save_meta           = True,
                    save_plot           = True,
                    plot_path           = os.path.join(
                        plot_dir, _plot_filename(pdb_id, resname, chain, resseq, ext)
                    ),
                )
                drawn += 1
            except Exception as e:
                errors.append(f"{pdb_id}_{resname}_{chain}_{resseq}: {e}")
    return drawn, errors


def _sample_rows_for_plots(summary_path: str, n: int, seed: int) -> List[dict]:
    """
    n rows drawn uniformly without replacement from summary_path's status=="ok"
    rows (n == -1 -> every ok row). Error rows are excluded: they carry no
    resolved mask, so there is nothing to draw. Rows are sorted before sampling
    so the draw is reproducible from the seed regardless of the order the
    parallel masking pass happened to append them in.
    """
    with open(summary_path, newline="", encoding="utf-8") as f:
        ok_rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]

    ok_rows.sort(key=lambda r: (r["pdb_id"], r["resname"], r["chain"] or "", r["resseq"] or ""))
    if n == -1 or n >= len(ok_rows):
        return ok_rows
    return random.Random(seed).sample(ok_rows, n)


def run_plot_pass(
    summary_path: str,
    pdb_root: str,
    xml_root: str,
    output_dir: str,
    include_types: List[str],
    mask_non_attractive: bool,
    plot_sample_n: int,
    plot_seed: int,
    workers: Optional[int] = None,
) -> int:
    """
    D9: draw interaction figures for plot_sample_n binding sites sampled from a
    finished summary CSV (-1 = all of them). Returns the number of figures drawn.
    """
    if plot_sample_n == 0:
        return 0
    if not os.path.isfile(summary_path):
        print(f"  ⚠️  No summary CSV at {summary_path} — skipping plot pass.")
        return 0

    rows = _sample_rows_for_plots(summary_path, plot_sample_n, plot_seed)
    if not rows:
        print("  ⚠️  No successful rows in the summary CSV — nothing to plot.")
        return 0

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    by_id: Dict[str, List[Tuple[str, Optional[str], Optional[int]]]] = {}
    for r in rows:
        resseq = int(r["resseq"]) if (r.get("resseq") or "").strip() else None
        by_id.setdefault(r["pdb_id"], []).append(
            (r["resname"], (r.get("chain") or "").strip() or None, resseq)
        )

    workers = workers if workers and workers > 0 else _default_worker_count()
    label = "ALL" if plot_sample_n == -1 else str(plot_sample_n)
    print(f"\n  Plot pass: {len(rows)} binding site(s) (requested {label}, "
          f"seed={plot_seed}) across {len(by_id)} structure(s) -> {plot_dir}")

    drawn, errors = 0, []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                plot_pdb_id, pdb_id, sites, pdb_root, xml_root, output_dir,
                plot_dir, include_types, mask_non_attractive,
            ): pdb_id
            for pdb_id, sites in by_id.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                            desc="Plotting sampled sites", unit="pdb"):
            try:
                n_drawn, errs = future.result()
            except Exception as e:
                n_drawn, errs = 0, [f"{futures[future]}: {e}"]
            drawn += n_drawn
            errors.extend(errs)

    print(f"  {drawn}/{len(rows)} figure(s) drawn.")
    for err in errors[:10]:
        print(f"    ⚠️  plot failed — {err}")
    if len(errors) > 10:
        print(f"    ... and {len(errors) - 10} more plot failure(s).")
    return drawn


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
    workers: Optional[int] = None,
    resume_from_csv: Optional[str] = None,
    retry_errors: bool = False,
    plot_sample_n: Optional[int] = None,
    plot_seed: int = 42,
) -> str:
    """
    Select PDB IDs (either a random sample of size n, or — sampling_mode="all" —
    every available PDB/XML pair), mask every binding site PLIP found in each,
    and write one aggregated summary CSV. Returns the summary CSV path.

    process_pdb_id() calls are dispatched across `workers` OS processes (D8);
    the main process is the sole writer and flushes+fsyncs the summary CSV
    after each PDB ID completes (in completion order, not submission order)
    so a run over the full 220k+ dataset can be safely interrupted. If
    resume=True and the summary CSV already exists, PDB IDs already recorded
    in it are skipped (D2: skip already-processed ids, not already-processed
    rows — process_pdb_id() writes all rows for an id in one call, so
    "present in the CSV" implies "fully processed").

    resume_from_csv points resume at an arbitrary existing summary CSV, which
    then also becomes the append target (D7). retry_errors re-queues ids that
    only appear as failures. plot_sample_n (default: config.MASK_CALC_SAVE_PLOTS
    read tri-state) draws verification figures for that many sampled binding
    sites once the masking pass finishes (D9).
    """
    include_types = include_types or list(config.INCLUDE_TYPES)
    os.makedirs(output_dir, exist_ok=True)

    if resume_from_csv and not resume:
        raise ValueError(
            "resume_from_csv is meaningless with resume=False (--no-resume would "
            "overwrite the very CSV you asked to continue from). Drop one of them."
        )
    if plot_sample_n is None:
        plot_sample_n = resolve_plot_sample_n()

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

    summary_path = resume_from_csv or os.path.join(output_dir, SUMMARY_CSV_NAME)
    if resume_from_csv and not os.path.isfile(resume_from_csv):
        raise FileNotFoundError(
            f"--resume-from CSV does not exist: {resume_from_csv}. Refusing to start "
            f"from scratch under a path that was meant to be resumed — check the path."
        )

    has_header = _check_summary_header(summary_path)

    completed: set = set()
    if resume:
        if retry_errors:
            completed, dropped = _drop_error_ids(summary_path)
            if dropped:
                print(f"  Retry-errors: dropped {dropped} row(s) from failed ids in "
                      f"{summary_path} — those structures will be reprocessed.")
        else:
            completed = _load_completed_ids(summary_path)
        if completed:
            before = len(picked)
            picked = [pid for pid in picked if pid not in completed]
            print(f"  Resume: {before - len(picked)} of {before} already in {summary_path} — skipping them.")
    file_mode = "a" if (resume and has_header) else "w"

    workers = workers if workers and workers > 0 else _default_worker_count()

    if not picked:
        print("  Nothing left to process (all picked ids already completed).")
        if file_mode == "w":
            with open(summary_path, file_mode, newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    else:
        print(f"  Workers: {workers} parallel process(es).")
        with open(summary_path, file_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if file_mode == "w":
                writer.writeheader()

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_pdb_id, pdb_id, pdb_root, xml_root, output_dir,
                        include_types, mask_non_attractive,
                    ): pdb_id
                    for pdb_id in picked
                }
                try:
                    for future in tqdm(as_completed(futures), total=len(futures),
                                        desc="Masking PDB structures", unit="pdb"):
                        pdb_id = futures[future]
                        try:
                            rows = future.result()
                        except Exception as e:
                            rows = [_error_row(pdb_id, "", None, None, mask_non_attractive, e)]
                        writer.writerows(rows)
                        f.flush()
                        os.fsync(f.fileno())
                except KeyboardInterrupt:
                    print("\n  Interrupted — already-completed rows are saved; "
                          "cancelling remaining pending work (rerun to resume)...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    with open(summary_path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    n_ok = sum(1 for r in all_rows if r["status"] == "ok")
    print(f"  {n_ok}/{len(all_rows)} binding sites masked successfully (cumulative across resumes).")
    print(f"  Summary CSV: {summary_path}")

    run_plot_pass(
        summary_path=summary_path, pdb_root=pdb_root, xml_root=xml_root,
        output_dir=output_dir, include_types=include_types,
        mask_non_attractive=mask_non_attractive,
        plot_sample_n=plot_sample_n, plot_seed=plot_seed, workers=workers,
    )
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
    p.add_argument("--resume-from", type=str, default=None,
                    help="Resume from (and APPEND to) this existing summary CSV instead of "
                         "the one in --output-dir. Its pdb_ids are skipped; its header must "
                         "match this scripts columns. Default: config.STAGE1B_RESUME_FROM_CSV.")
    p.add_argument("--retry-errors", action="store_true", default=None,
                    help="Reprocess pdb_ids that are present in the summary CSV only as "
                         "failures: their old rows are dropped from the CSV and the ids "
                         "re-queued. Default: config.STAGE1B_RETRY_ERRORS.")
    p.add_argument("--plot-sample-n", type=int, default=None,
                    help="Number of randomly sampled binding sites to draw 2D interaction "
                         "figures for after masking; 0 = none, -1 = every one. Default: "
                         "config.MASK_CALC_SAVE_PLOTS read tri-state (False=0, True=-1, int=N).")
    p.add_argument("--plot-seed", type=int, default=None,
                    help="Seed for the plot sample (default: config.STAGE1B_PLOT_SEED).")
    p.add_argument("--plot-only", action="store_true",
                    help="Skip masking entirely; just draw the sampled figures from an "
                         "existing summary CSV (--resume-from, or --output-dir).")
    p.add_argument("--workers", type=int, default=None,
                    help="Number of parallel worker processes (default: auto-detected from "
                         "$SLURM_CPUS_PER_TASK / CPU affinity — see D8).")
    p.add_argument("--test", action="store_true", help="Run the self-test and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    print("\n" + "=" * 60)
    print("STAGE 1b: LARGE-SCALE PLIP MASK CALCULATION")
    print("=" * 60)

    args = _parse_args(argv)

    sampling_mode = (
        args.sampling_mode
        or getattr(config, "STAGE1B_SAMPLING_MODE", None)
        or _ask_sampling_mode()
    )
    if sampling_mode not in ("n", "all"):
        raise ValueError(
            f"Invalid sampling mode {sampling_mode!r} (config.STAGE1B_SAMPLING_MODE?) "
            f"- expected 'n' or 'all'."
        )
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
    resume_from_csv = args.resume_from or getattr(config, "STAGE1B_RESUME_FROM_CSV", "") or None
    retry_errors = (
        args.retry_errors if args.retry_errors is not None
        else bool(getattr(config, "STAGE1B_RETRY_ERRORS", False))
    )
    plot_sample_n = (
        args.plot_sample_n if args.plot_sample_n is not None else resolve_plot_sample_n()
    )
    plot_seed = (
        args.plot_seed if args.plot_seed is not None
        else int(getattr(config, "STAGE1B_PLOT_SEED", 42))
    )
    workers = args.workers if args.workers is not None else _prompt_int(
        "Number of parallel worker processes", _default_worker_count()
    )
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
  Resume           : {resume}{" (from " + resume_from_csv + ")" if resume_from_csv else ""}
  Retry errors     : {retry_errors}
  Plot sample      : {"ALL binding sites" if plot_sample_n == -1 else ("none" if plot_sample_n == 0 else f"{plot_sample_n} (seed={plot_seed})")}
  Workers          : {workers}
""")

    if args.plot_only:
        summary_path = resume_from_csv or os.path.join(output_dir, SUMMARY_CSV_NAME)
        print(f"  --plot-only: masking skipped, plotting from {summary_path}")
        run_plot_pass(
            summary_path=summary_path, pdb_root=pdb_root, xml_root=xml_root,
            output_dir=output_dir, include_types=include_types,
            mask_non_attractive=mask_non_attractive,
            plot_sample_n=plot_sample_n, plot_seed=plot_seed, workers=workers,
        )
        print("\n✅ Stage 1b plot pass complete.")
        return

    run_large_scale_plip_masking(
        n=n, seed=seed, pdb_root=pdb_root, xml_root=xml_root,
        output_dir=output_dir, mask_non_attractive=mask_non_attractive,
        include_types=include_types, sampling_mode=sampling_mode, resume=resume,
        workers=workers, resume_from_csv=resume_from_csv, retry_errors=retry_errors,
        plot_sample_n=plot_sample_n, plot_seed=plot_seed,
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
            output_dir=td, mask_non_attractive=True, workers=2,
        )
        assert os.path.isfile(summary_path)
        with open(summary_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3, f"Expected 3 summary rows (one per binding site), got {len(rows)}"
        for row in rows:
            assert row["pdb_id"] == "100d"
            assert row["status"] in ("ok", "error")

        # D4: the masking pass writes no .meta.json and no figures.
        strays = glob.glob(os.path.join(td, "*.meta.json")) + glob.glob(
            os.path.join(td, "**", "*2d_interactions*"), recursive=True)
        assert not strays, f"Masking pass should write no per-ligand artifacts, found {strays}"

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

        # Header guard: appending under mismatched columns must abort, not corrupt.
        bad_csv = os.path.join(td, "bad_header.csv")
        with open(bad_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["pdb_id", "something_else"])
        try:
            _check_summary_header(bad_csv)
        except RuntimeError:
            pass
        else:
            raise AssertionError("_check_summary_header should reject a mismatched header")

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

    # --resume-from: an arbitrary CSV elsewhere is both skipped-from and appended to.
    with tempfile.TemporaryDirectory() as td:
        carried = os.path.join(td, "carried_over.csv")
        with open(carried, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerow(_error_row("100d", "C", "A", 1, True, ValueError("stale failure")))

        out_dir = os.path.join(td, "out")
        returned = run_large_scale_plip_masking(
            n=1, seed=0, pdb_root=fixture_pdb_root, xml_root=fixture_xml_root,
            output_dir=out_dir, mask_non_attractive=True, sampling_mode="all",
            resume=True, resume_from_csv=carried, plot_sample_n=0,
        )
        assert returned == carried, f"Should resume into the given CSV, got {returned}"
        assert not os.path.isfile(os.path.join(out_dir, SUMMARY_CSV_NAME)), (
            "--resume-from must not also write the default summary CSV")
        with open(carried, newline="", encoding="utf-8") as f:
            carried_rows = list(csv.DictReader(f))
        assert len(carried_rows) == 1, (
            f"100d was already in the carried CSV, so nothing should be appended; "
            f"got {len(carried_rows)} rows")

        # ...and retry_errors re-queues it, REPLACING the stale error row rather
        # than duplicating the id.
        run_large_scale_plip_masking(
            n=1, seed=0, pdb_root=fixture_pdb_root, xml_root=fixture_xml_root,
            output_dir=out_dir, mask_non_attractive=True, sampling_mode="all",
            resume=True, resume_from_csv=carried, retry_errors=True, plot_sample_n=0,
        )
        with open(carried, newline="", encoding="utf-8") as f:
            retried_rows = list(csv.DictReader(f))
        assert len(retried_rows) == 3, (
            f"retry_errors should replace the 1 stale row with 3 fresh ones, "
            f"got {len(retried_rows)}")
        assert not any(r["error"] == "stale failure" for r in retried_rows), (
            "stale error row survived retry_errors")

    # Plot pass (D9): exactly N figures, not one per binding site.
    with tempfile.TemporaryDirectory() as td:
        summary = run_large_scale_plip_masking(
            n=1, seed=0, pdb_root=fixture_pdb_root, xml_root=fixture_xml_root,
            output_dir=td, mask_non_attractive=True, sampling_mode="all",
            plot_sample_n=0,
        )
        with open(summary, newline="", encoding="utf-8") as f:
            n_ok = sum(1 for r in csv.DictReader(f) if r["status"] == "ok")

        if n_ok >= 2:
            drawn = run_plot_pass(
                summary_path=summary, pdb_root=fixture_pdb_root, xml_root=fixture_xml_root,
                output_dir=td, include_types=list(config.INCLUDE_TYPES),
                mask_non_attractive=True, plot_sample_n=2, plot_seed=0, workers=2,
            )
            figs = glob.glob(os.path.join(td, "plots", "*2d_interactions*"))
            assert drawn == 2, f"Expected exactly 2 figures drawn, got {drawn}"
            assert len(figs) == 2, f"Expected exactly 2 figure files, got {len(figs)}"
            # Sampled ligands keep their .meta.json (the answer key for the figure).
            assert glob.glob(os.path.join(td, "*.meta.json")), (
                "Plotted ligands should keep their .meta.json")
        else:
            print(f"  (skipped plot-count assertion: only {n_ok} ok row(s) in fixture)")

        # plot_sample_n=0 draws nothing.
        assert run_plot_pass(
            summary_path=summary, pdb_root=fixture_pdb_root, xml_root=fixture_xml_root,
            output_dir=td, include_types=list(config.INCLUDE_TYPES),
            mask_non_attractive=True, plot_sample_n=0, plot_seed=0,
        ) == 0

    # Tri-state reading of config.MASK_CALC_SAVE_PLOTS (D9).
    _saved = getattr(config, "MASK_CALC_SAVE_PLOTS", True)
    try:
        for value, expected in ((False, 0), (0, 0), (True, -1), (5, 5), (-3, 0)):
            config.MASK_CALC_SAVE_PLOTS = value
            got = resolve_plot_sample_n()
            assert got == expected, f"MASK_CALC_SAVE_PLOTS={value!r} -> {got}, expected {expected}"
    finally:
        config.MASK_CALC_SAVE_PLOTS = _saved

    print("✅ Stage 1b large-scale PLIP masking self-test passed.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main()

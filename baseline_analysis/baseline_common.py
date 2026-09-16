# -*- coding: utf-8 -*-
"""
baseline_common.py
==================
Shared plumbing for the vanilla-ChemBERTa baseline analysis. Nothing in here
generates, masks or docks anything by itself -- it is the small amount of glue
the four baseline scripts have in common:

  * where the bundle lives and how it is laid out          (bundle_dir, paths)
  * the three masking arms and their display names          (ARMS, ARM_LABEL)
  * reading the Stage 1b PLIP-- corpus out of a dir/CSV/tar (read_negative_rows)
  * turning one corpus row into the three atom pools        (pools_for_parent)
  * applying the <=BASELINE_MASK_PERCENT% budget            (build_mask)
  * provenance: sha256 + git rev + a manifest               (write_manifest)

WHY THE ARMS ARE BUILT THE WAY THEY ARE
----------------------------------------
Stage 1 (stage1_mask_calculation.run_pipeline) resolves, for one binding site,
the set of ligand atoms PLIP reports as contacting the protein ("attractive").
Its mode 2 then masks the COMPLEMENT of that set:

    masked_indices = set(range(mol.GetNumAtoms())) - attractive_indices

so a mode-2 corpus row already encodes BOTH pools exactly:

    plip_neg  pool = the row's masked_atom_indices      (non-interacting atoms)
    plip_pos  pool = range(n_atoms) - masked_atom_indices  (interacting atoms)
    random    pool = range(n_atoms)                      (every atom)

That is why one PLIP-- corpus (config.STAGE1B_PLIP_NEGATIVE_MASK_DIR) feeds both
PLIP arms and no fresh PLIP run is needed. build_masks.py --verify-plip-pos
re-derives the positive pool the long way (an actual Stage 1b --mode 1 call on
the same PDB + XML) and asserts the two agree, so the shortcut is checked rather
than assumed.

THE BUDGET IS THE SAME FOR EVERY ARM
-------------------------------------
build_mask delegates to stage9_masked_property_finetune._remask_from_pool -- the
SAME function Stage 9 trains on and Stage 9a baselines with, not a reimplementation.
Its rule is floor(percent/100 * n_bpe_tokens) atom indices drawn from the arm's
pool, and min(target, len(pool)) when the pool is smaller. So:

  * no arm can exceed BASELINE_MASK_PERCENT% of the parent's BPE tokens, and
  * PLIP++ (whose pool is typically a handful of atoms) masks its whole pool,
    which is the intended reading of "keep at most 15% of the tokens masked".

Masking is token-level (mask_atoms_in_smiles_token_level), so one BPE token
covering several masked atoms collapses to ONE <mask>: the realised mask-token
count is <= the atom count drawn, never above. masks.csv records both.
"""
from __future__ import annotations

import fnmatch
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# The repo root is the import root for config / stageN modules.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config  # noqa: E402

# ── The three arms ───────────────────────────────────────────────────────────
# "plip_pos" = PLIP ++ : mask the atoms PLIP says DO contact the protein.
# "plip_neg" = PLIP -- : mask the atoms PLIP says do NOT contact the protein.
# "random"   =          mask atoms drawn uniformly from the whole molecule.
ARMS: Tuple[str, ...] = ("plip_pos", "plip_neg", "random")

ARM_LABEL = {
    "plip_pos":      "PLIP ++ (interacting atoms masked)",
    "plip_neg":      "PLIP -- (non-interacting atoms masked)",
    "random":        "Random (uniform over all atoms)",
    "parent":        "Parent (crystal ligand)",
    "parent_redock": "Parent redock (reference)",
}

# Reference arm name used in the docking table for the redocked crystal ligand.
PARENT_ARM = "parent_redock"

SUMMARY_CSV_NAME = "stage1b_large_scale_plip_mask_summary.csv"


# ════════════════════════════════════════════════════════════════════════════
#  BUNDLE LAYOUT
# ════════════════════════════════════════════════════════════════════════════

def bundle_dir(explicit: Optional[str] = None) -> str:
    """
    Root of the baseline bundle. --out-dir wins, then config.BASELINE_DIR, then
    baseline_analysis/out next to this file (the local default, so the selection
    step works on a laptop with no pipeline output tree).
    """
    if explicit:
        return os.path.abspath(explicit)
    cfg = getattr(config, "BASELINE_DIR", "")
    if cfg:
        return os.path.abspath(cfg)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def paths(root: str) -> Dict[str, str]:
    """Every file the baseline scripts read or write, in one place."""
    return {
        "root":          root,
        "selection":     os.path.join(root, "selection.csv"),
        "candidates":    os.path.join(root, "candidates_all.csv"),
        "overview":      os.path.join(root, "overview.txt"),
        "complexes":     os.path.join(root, "complexes"),
        "masks":         os.path.join(root, "masks.csv"),
        "mask_pools":    os.path.join(root, "mask_pools.json"),
        "manifest":      os.path.join(root, "manifest.json"),
        "predictions":   os.path.join(root, "predictions.csv"),
        "docking":       os.path.join(root, "docking_summary.csv"),
        "docking_work":  os.path.join(root, "docking"),
        "per_molecule":  os.path.join(root, "per_molecule.csv"),
        "arm_summary":   os.path.join(root, "arm_summary.csv"),
        "complex_delta": os.path.join(root, "per_complex_delta.csv"),
        "stats":         os.path.join(root, "summary_stats.csv"),
        "figures":       os.path.join(root, "figures"),
        "results":       os.path.join(root, "RESULTS.md"),
    }


def complex_dir(root: str, complex_id: str) -> str:
    """Per-complex working directory; the id's ':' is not path-safe."""
    return os.path.join(paths(root)["complexes"], complex_id.replace(":", "_"))


def parse_complex_id(complex_id: str) -> Tuple[str, str, str, str]:
    """'1ERR:RAL:B:600' -> ('1err', 'RAL', 'B', '600'). PDB id lower-cased."""
    parts = str(complex_id).split(":")
    if len(parts) != 4:
        raise ValueError(
            f"complex id must be PDB:LIG:CHAIN:POS, got {complex_id!r}"
        )
    pdb_id, resname, chain, resseq = parts
    return pdb_id.strip().lower(), resname.strip(), chain.strip(), resseq.strip()


def pdb_gz_path(pdb_root: str, pdb_id: str) -> str:
    """Local PDB mirror layout: <root>/<id[1:3]>/pdb<id>.ent.gz (as Stage 1b)."""
    pdb_id = pdb_id.lower()
    return os.path.join(pdb_root, pdb_id[1:3], f"pdb{pdb_id}.ent.gz")


def plip_xml_path(xml_root: str, pdb_id: str) -> str:
    """Pre-computed PLIP XML, flat: <root>/pdb<id>.xml (as Stage 1b)."""
    return os.path.join(xml_root, f"pdb{pdb_id.lower()}.xml")


# ════════════════════════════════════════════════════════════════════════════
#  READING THE PLIP-- CORPUS
# ════════════════════════════════════════════════════════════════════════════

def _iter_summary_streams(location: str) -> Iterable[Tuple[str, io.TextIOBase]]:
    """
    Yield (name, text-stream) for every Stage 1b summary CSV at `location`,
    which may be the CSV itself, a directory holding it, or a .tar/.tar.gz.
    Streaming (not slurping) matters: the full corpus CSV is ~100 MB and
    holds ~585k rows, of which a baseline run wants a few dozen.
    """
    if not location:
        return
    loc = os.path.abspath(location)

    if os.path.isfile(loc) and loc.lower().endswith(".csv"):
        with open(loc, newline="", encoding="utf-8", errors="replace") as f:
            yield loc, f
        return

    if os.path.isfile(loc) and (".tar" in os.path.basename(loc).lower()):
        with tarfile.open(loc, "r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if not fnmatch.fnmatch(os.path.basename(member.name), SUMMARY_CSV_NAME):
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                yield member.name, io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        return

    if os.path.isdir(loc):
        for csv_path in sorted(glob.glob(os.path.join(loc, SUMMARY_CSV_NAME))):
            with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
                yield csv_path, f
        return


def read_negative_rows(
    location: str,
    wanted: Sequence[Tuple[str, str, str, str]],
) -> Dict[Tuple[str, str, str, str], dict]:
    """
    Pull just the `wanted` (pdb_id, resname, chain, resseq) rows out of the
    Stage 1b PLIP-- corpus. Keys are matched with the pdb_id lower-cased and
    every field str()-ed, because the corpus stores resseq as an int-looking
    string and metadata.tsv as text.

    status != "ok" rows are skipped: they carry no SMILES and no pool.
    """
    import csv as _csv

    want = {(str(a).lower(), str(b), str(c), str(d)) for a, b, c, d in wanted}
    found: Dict[Tuple[str, str, str, str], dict] = {}
    if not want:
        return found

    for name, stream in _iter_summary_streams(location):
        reader = _csv.DictReader(stream)
        if not reader.fieldnames or "masked_atom_indices" not in reader.fieldnames:
            print(f"  [corpus] {name}: not a Stage 1b summary CSV, skipped")
            continue
        for row in reader:
            if row.get("status") != "ok":
                continue
            key = (
                str(row.get("pdb_id", "")).lower(),
                str(row.get("resname", "")),
                str(row.get("chain", "")),
                str(row.get("resseq", "")),
            )
            if key in want and key not in found:
                found[key] = row
                if len(found) == len(want):
                    return found
    return found


# ════════════════════════════════════════════════════════════════════════════
#  POOLS + BUDGET
# ════════════════════════════════════════════════════════════════════════════

def n_heavy_atoms(smiles: str) -> int:
    """Atom count RDKit sees, which is what the Stage 1 pools are indexed by."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse parent SMILES: {smiles!r}")
    return mol.GetNumAtoms()


def pools_for_parent(smiles: str, negative_pool: Sequence[int]) -> Dict[str, List[int]]:
    """
    The three arms' candidate atom pools for one parent, derived from a Stage 1b
    PLIP-- row (see this module's docstring for why the complement is exact).

    Raises if the corpus pool is not a subset of the molecule's atom indices --
    that would mean the row and the SMILES disagree, and silently intersecting
    them would produce a quietly wrong "interacting" set.
    """
    n = n_heavy_atoms(smiles)
    neg = sorted({int(i) for i in negative_pool})
    out_of_range = [i for i in neg if i < 0 or i >= n]
    if out_of_range:
        raise ValueError(
            f"PLIP-- pool has atom indices outside the parent molecule "
            f"(n_atoms={n}, offending={out_of_range[:8]})"
        )
    pos = [i for i in range(n) if i not in set(neg)]
    return {"plip_pos": pos, "plip_neg": neg, "random": list(range(n))}


def mask_budget(smiles: str, tokenizer, percent: float) -> Tuple[int, int]:
    """(n_bpe_tokens, budget) where budget = floor(percent% * n_bpe_tokens)."""
    n_tokens = len(tokenizer(smiles, add_special_tokens=False)["input_ids"])
    return n_tokens, int(percent / 100.0 * n_tokens)


def build_mask(
    smiles: str,
    pool: Sequence[int],
    percent: float,
    tokenizer,
    seed_key: str,
    base_seed: int,
) -> Tuple[str, Optional[str], List[int]]:
    """
    Mask at most `percent`% of the parent's BPE TOKENS, drawing atoms from `pool`.

    Returns (masked_smiles, skip_reason, sampled_atom_indices). skip_reason is
    None on success; otherwise masked_smiles is "" and the caller records the
    reason ("empty_pool" / "no_mask_at_percent" / "invalid_parent" /
    "over_length" / "mask_failed") instead of aborting -- PDB-derived parents
    include SMILES RDKit refuses outright, and one bad row must not kill a run.

    WHY THIS DOES NOT JUST CALL Stage 9's _remask_from_pool
    --------------------------------------------------------
    Stage 9 draws floor(percent% * n_tokens) ATOM indices and masks them. That is
    a budget on atoms, and it is NOT the same as a budget on tokens: masking is
    token-level (mask_atoms_in_smiles_token_level replaces every BPE token that
    overlaps a masked atom), and ONE atom can span SEVERAL tokens -- a bracket
    atom like [C@@H] or [S@](=O) is routinely 2-4 BPE tokens. Measured on this
    selection, Stage 9's rule produced 15 <mask> tokens on an 83-token parent
    whose 15% budget is 12, i.e. ~18% of the sequence masked.

    The requirement for this baseline is "keep at most 15% of the tokens masked",
    so the budget is enforced on the REALISED mask-token count instead:

      1. draw order is a deterministic shuffle of the pool, seeded exactly as
         Stage 9 seeds its sample (_remask_seed(base_seed, seed_key)), so the
         same complex/arm/seed always yields the same mask;
      2. atoms are added one at a time, and an atom is KEPT only if the string it
         produces still has <= budget mask tokens -- an atom that would overflow
         is skipped, not stopped on, so a cheap single-token atom can still fill
         the remaining budget;
      3. the count is read off the finished string (the thing the model actually
         sees), never predicted.

    The consequence is that all three arms are comparable on the axis that
    matters -- how much of the sequence the model must reconstruct -- and
    build_masks.assert_budget_respected re-checks it independently.
    """
    import random as _random

    from rdkit import Chem

    import stage9_masked_property_finetune as s9
    from stage1_mask_calculation import mask_atoms_in_smiles_token_level

    pool_list = [int(i) for i in pool]
    if not pool_list:
        return "", "empty_pool", []
    if Chem.MolFromSmiles(smiles) is None:
        return "", "invalid_parent", []

    n_tokens, budget = mask_budget(smiles, tokenizer, percent)
    if n_tokens + 2 > s9.MAX_MODEL_TOKENS:        # +2 for <s> / </s>
        return "", "over_length", []
    if budget <= 0:
        return "", "no_mask_at_percent", []

    order = list(pool_list)
    _random.Random(s9._remask_seed(base_seed, seed_key)).shuffle(order)

    chosen: List[int] = []
    masked = ""
    for atom in order:
        trial = sorted(chosen + [atom])
        try:
            cand = mask_atoms_in_smiles_token_level(smiles, trial, tokenizer)
        except Exception:
            continue                                # this atom is unmaskable; try the next
        if not cand or tokenizer.mask_token not in cand:
            continue
        if count_mask_tokens(cand, tokenizer) <= budget:
            chosen, masked = trial, cand
        if len(chosen) == len(pool_list):
            break

    if not chosen or not masked:
        return "", "mask_failed", []
    # The masked string is what the model sees and can be LONGER in tokens than
    # the parent (each <mask> is its own token), so it gets its own check.
    if len(tokenizer(masked, add_special_tokens=True)["input_ids"]) > s9.MAX_MODEL_TOKENS:
        return "", "over_length", []
    return masked, None, chosen


def count_mask_tokens(masked_smiles: str, tokenizer) -> int:
    """How many <mask> tokens the model will actually see."""
    tok = tokenizer.mask_token
    return masked_smiles.count(tok) if tok else 0


# ════════════════════════════════════════════════════════════════════════════
#  PROVENANCE
# ════════════════════════════════════════════════════════════════════════════

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def write_manifest(root: str, section: str, payload: dict) -> str:
    """
    Merge `payload` into manifest.json under `section`, creating the file if
    needed. One manifest accumulates every step's provenance (selection, masks,
    generation, docking) so re-running the baseline with a different model can
    be checked against it byte for byte.
    """
    path = paths(root)["manifest"]
    data = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data[section] = payload
    data.setdefault("_repo", {})["git_rev"] = git_rev()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def baseline_settings() -> dict:
    """The config knobs every step must agree on, recorded in the manifest."""
    return {
        "chemberta_model":  getattr(config, "CHEMBERTA_MODEL", ""),
        "mask_percent":     getattr(config, "BASELINE_MASK_PERCENT", getattr(config, "MASK_PERCENT", 15)),
        "n_complexes":      getattr(config, "BASELINE_N_COMPLEXES", 24),
        "n_seeds":          getattr(config, "BASELINE_N_SEEDS", 5),
        "mask_seed":        getattr(config, "BASELINE_MASK_SEED", 42),
        "gen_seed":         getattr(config, "BASELINE_GEN_SEED", 1000),
        "top_k":            getattr(config, "BASELINE_TOP_K", 20),
        "temperature":      getattr(config, "BASELINE_TEMPERATURE", 1.2),
        "token_level_masking": bool(getattr(config, "TOKEN_LEVEL_MASKING", False)),
        "bpe_mask_adapter":    bool(getattr(config, "BPE_MASK_ADAPTER_ENABLED", True)),
        "config_platform":  getattr(config, "CONFIG_PLATFORM", "?"),
    }


# ════════════════════════════════════════════════════════════════════════════
#  SMALL SHARED CSV HELPERS
# ════════════════════════════════════════════════════════════════════════════

def read_csv_rows(path: str) -> List[dict]:
    import csv as _csv

    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def write_csv_rows(path: str, rows: List[dict], fields: Sequence[str]) -> None:
    import csv as _csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)

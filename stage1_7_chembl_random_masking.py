# -*- coding: utf-8 -*-
"""
stage1_7_chembl_random_masking.py
=================================
Stage 1.7 of the pipeline:

    ChEMBL BRD4 cache  →  random-mask .json files (one per molecule)

For every BRD4-active molecule in the ChEMBL cache, samples a random subset
of atom indices (fraction of heavy atoms) and writes a JSON in the same
format as Stage 1 / 1.5 so Stage 1.9 can train on the enlarged dataset.

ChEMBL data source
------------------
  • If config.CHEMBL_CACHE_PATH exists  → read it (auto, no prompt).
  • If cache is missing                 → fetch via Stage 5 helpers using
    config.CHEMBL_PCHEMBL_MIN and write the cache for reuse in Stage 5.

Mask fraction (Ambiguity 7)
---------------------------
  • If config.CHEMBL_MASK_FRACTION is set → use that exact fraction.
  • Otherwise prompt: enter a fraction (0–1) or press Enter for the
    PDB-derived mean mask rate from Stage 1 (7D).

Filters (Ambiguity 9A)
----------------------
  RDKit-valid SMILES and min_heavy_atoms ≥ 7 (same as Stage 5).

Resumable
---------
  A molecule whose JSON already exists in CHEMBL_MASK_OUTDIR is skipped (not
  re-masked). Random masks use a per-molecule deterministic seed derived from
  (config.RANDOM_MASK_SEED, chembl_id), so a resumed run produces exactly the
  same masks as a fresh full run. Note: existing files are kept even if you
  change the mask fraction — delete the output folder to force a full rebuild.

Run
---
    python stage1_7_chembl_random_masking.py

Prerequisites
-------------
  Stage 1 recommended (for 7D fallback).  ChEMBL cache or network for fetch.
  Run before stage1_9_LLM_RDkit_policy_training.py.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from typing import List, Optional, Tuple

from rdkit import Chem

import config

from stage5_chembl_matching import load_br4_ligands, load_or_fetch_chembl

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else range(0)


CHEMBL_MIN_HEAVY_ATOMS = 7


def compute_pdb_mask_fraction(stage1_dir: str) -> float:
    """
    7D: mean of (len(masked_atom_indices) / n_atoms) across Stage-1 JSONs.
    """
    rates: List[float] = []
    for path in glob.glob(os.path.join(stage1_dir, "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            indices = meta.get("masked_atom_indices") or []
            if not indices:
                continue
            mol = Chem.MolFromSmiles(meta["smiles"])
            if mol is None:
                continue
            n_atoms = mol.GetNumAtoms()
            if n_atoms <= 0:
                continue
            rates.append(len(indices) / n_atoms)
        except Exception:
            pass

    if not rates:
        raise ValueError(
            f"No usable Stage-1 JSONs in {stage1_dir} for PDB-derived mask rate."
        )
    return sum(rates) / len(rates)


def resolve_mask_fraction(stage1_dir: str) -> Tuple[float, str]:
    """
    Return (fraction, source_label).

    Uses config.CHEMBL_MASK_FRACTION when set; otherwise prompts or 7D.
    """
    if hasattr(config, "CHEMBL_MASK_FRACTION"):
        frac = getattr(config, "CHEMBL_MASK_FRACTION")
        if frac is not None:
            frac_f = float(frac)
            if not 0.0 < frac_f <= 1.0:
                raise ValueError(
                    f"CHEMBL_MASK_FRACTION must be in (0, 1], got {frac_f}"
                )
            return frac_f, "config.CHEMBL_MASK_FRACTION"

    while True:
        raw = input(
            "  ChEMBL mask fraction (0–1), or Enter for PDB-derived rate [7D]: "
        ).strip()
        if not raw:
            try:
                frac_f = compute_pdb_mask_fraction(stage1_dir)
                return frac_f, "PDB-derived (7D)"
            except ValueError as e:
                print(f"  ⚠️  {e}")
                print("  Please enter an explicit fraction (e.g. 0.25).")
                continue
        try:
            frac_f = float(raw)
            if 0.0 < frac_f <= 1.0:
                return frac_f, "user input"
            print("  Fraction must be in (0, 1].")
        except ValueError:
            print("  Invalid number — try again (e.g. 0.25).")


def load_chembl_molecules() -> List[Tuple[str, str]]:
    """
    Load BRD4 ChEMBL molecules from cache (auto) or fetch + cache if missing.
    Returns list of (chembl_id, canonical_smiles) after heavy-atom filtering.
    """
    cache_path = config.CHEMBL_CACHE_PATH
    pchembl_min = getattr(config, "CHEMBL_PCHEMBL_MIN", 5.0)

    if os.path.isfile(cache_path):
        print(f"  Using ChEMBL cache: {cache_path}")
    else:
        print(f"  Cache not found — fetching ChEMBL (pChEMBL ≥ {pchembl_min}) …")
        load_or_fetch_chembl(
            pchembl_min  = pchembl_min,
            cache_path   = cache_path,
            use_cache    = False,
        )

    ligands = load_br4_ligands(cache_path, min_heavy_atoms=CHEMBL_MIN_HEAVY_ATOMS)
    print(
        f"  {len(ligands)} ChEMBL molecules after filter "
        f"(RDKit-valid, ≥{CHEMBL_MIN_HEAVY_ATOMS} heavy atoms)."
    )
    return ligands


def mask_count_for_molecule(n_atoms: int, fraction: float) -> int:
    """Number of atoms to mask: round(fraction × n_atoms), at least 1, at most n_atoms."""
    if n_atoms <= 0:
        return 0
    n_mask = max(1, round(fraction * n_atoms))
    return min(n_mask, n_atoms)


def _molecule_seed(chembl_id: str, base_seed: int) -> int:
    """
    Deterministic per-molecule RNG seed, stable across processes and runs.

    Using a hash of (base_seed, chembl_id) makes each molecule's random mask
    independent of processing order. Skipping already-masked molecules on a
    resumed run therefore yields exactly the same masks as a fresh full run.
    (Python's built-in hash() is process-salted, so sha256 is used instead.)
    """
    digest = hashlib.sha256(f"{base_seed}:{chembl_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def generate_chembl_mask_json(
    chembl_id: str,
    smiles: str,
    fraction: float,
    output_path: str,
    base_seed: Optional[int] = None,
) -> dict:
    """
    Sample random atom indices and write one Stage-1.5-style JSON.

    The RNG is seeded per molecule via _molecule_seed(chembl_id, base_seed),
    so each molecule's mask is reproducible and order-independent.
    """
    if base_seed is None:
        base_seed = config.RANDOM_MASK_SEED

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for {chembl_id}: {smiles}")

    n_atoms = mol.GetNumAtoms()
    n_mask  = mask_count_for_molecule(n_atoms, fraction)
    if n_mask <= 0:
        raise ValueError(f"No maskable atoms for {chembl_id}")

    rng = random.Random(_molecule_seed(chembl_id, base_seed))
    indices = sorted(rng.sample(range(n_atoms), n_mask))

    meta = {
        "smiles":              smiles,
        "masked_atom_indices": indices,
        "ligand": {
            "resname": chembl_id,
            "chain":   "C",
            "resseq":  0,
        },
        "masking_mode":  "random_chembl",
        "source":        "stage1_7",
        "chembl_id":     chembl_id,
        "n_atoms":       n_atoms,
        "mask_fraction": fraction,
        "n_masks":       n_mask,
        "seed":          base_seed,
    }

    from bpe_mask_adapter import build_smiles_mask_json_fields
    meta.update(build_smiles_mask_json_fields(smiles, indices))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def run_stage1_7(
    output_dir: Optional[str] = None,
    stage1_dir: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Generate ChEMBL random-mask JSONs.

    Resumable: a molecule whose JSON already exists in output_dir is skipped
    (not re-masked). Combined with per-molecule deterministic seeding, a
    resumed run produces exactly the same masks as a fresh full run.

    Returns (written_count, invalid_count, already_existed_count).
    """
    output_dir = output_dir or config.CHEMBL_MASK_OUTDIR
    stage1_dir = stage1_dir or config.MASK_CALC_OUTDIR
    os.makedirs(output_dir, exist_ok=True)

    fraction, frac_source = resolve_mask_fraction(stage1_dir)
    print(f"  Mask fraction    : {fraction:.4f}  ({frac_source})")
    print(f"  RNG seed         : {config.RANDOM_MASK_SEED}")

    ligands = load_chembl_molecules()
    if not ligands:
        print("  No ChEMBL molecules available. Exiting.")
        return 0, 0, 0

    written, invalid, already = 0, 0, 0

    for chembl_id, smiles in tqdm(ligands, desc="ChEMBL masking", unit="mol"):
        out_path = os.path.join(output_dir, f"{chembl_id}.chembl.json")

        # Resume: skip molecules that were already masked in a previous run.
        if os.path.isfile(out_path):
            already += 1
            continue

        try:
            generate_chembl_mask_json(
                chembl_id   = chembl_id,
                smiles      = smiles,
                fraction    = fraction,
                output_path = out_path,
            )
            written += 1
        except ValueError as e:
            tqdm.write(f"  ⚠️  Skipped {chembl_id}: {e}")
            invalid += 1

    return written, invalid, already


def main() -> None:
    print("\n" + "=" * 60)
    print("STAGE 1.7: ChEMBL BRD4 RANDOM MASK GENERATION")
    print("=" * 60)
    print(f"""
  Reads BRD4-active molecules from:
    {config.CHEMBL_CACHE_PATH}
  (fetches with pChEMBL ≥ {getattr(config, 'CHEMBL_PCHEMBL_MIN', 5.0)} if cache missing)

  Writes one random-mask JSON per molecule to:
    {config.CHEMBL_MASK_OUTDIR}
""")

    written, invalid, already = run_stage1_7()

    print(
        f"\n  Stage 1.7 complete.  {written} written, "
        f"{already} already existed (skipped), {invalid} invalid (skipped)."
    )
    print(f"  Output: {config.CHEMBL_MASK_OUTDIR}")
    print("  Run stage1_9_LLM_RDkit_policy_training.py next.")


if __name__ == "__main__":
    main()

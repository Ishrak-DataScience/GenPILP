# -*- coding: utf-8 -*-
"""
stage1_5_random_masking.py
==========================
Stage 1.5 of the pipeline:

    Stage-1 .meta.json files  →  random-mask .meta.json files

For every ligand JSON produced by Stage 1, this stage produces a matching
JSON in RANDOM_MASK_OUTDIR with the SAME format but with atom indices chosen
UNIFORMLY AT RANDOM (no interaction awareness).

The random JSON contains exactly N random indices, where
    N = len(stage1_meta["masked_atom_indices"])
so both strategy lists are always the same length per ligand, making the
incremental plots in Stage 2 directly comparable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS (read before use)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1. A "valid maskable position" is any 0-based RDKit atom index
    (0 … mol.GetNumAtoms()-1). Hydrogen atoms ARE included if the
    SMILES has explicit Hs; implicit Hs are not separate atoms.
    This is consistent with how Stage 1 treats atom indices.

A2. N random indices are sampled WITHOUT replacement.
    If N > mol.GetNumAtoms() (edge case: more interaction-aware masks
    than heavy atoms — should not happen in practice), N is silently
    capped at mol.GetNumAtoms() and a warning is printed.

A3. Sampling is seeded with config.RANDOM_MASK_SEED so the random
    indices are reproducible across runs.

A4. The output JSON re-uses the "smiles" and "ligand" fields verbatim
    from the Stage-1 JSON so Stage 2 can link them by the ligand key.

A5. If a Stage-1 JSON has zero masked_atom_indices (no interactions
    found), the ligand is skipped with a warning — there is nothing to
    compare against.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage1_5_random_masking.py

Must be run AFTER stage1_mask_calculation.py.
Must be run BEFORE stage2_molecule_generation.py.

HOW TO TEST (quick sanity check)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from stage1_5_random_masking import generate_random_mask_json
    from rdkit import Chem
    import json, tempfile, os

    # Minimal Stage-1-style JSON
    stage1_data = {
        "smiles": "c1ccccc1",        # benzene, 6 atoms
        "masked_atom_indices": [0, 2, 4],   # N=3
        "ligand": {"resname": "BNZ", "chain": "A", "resseq": 1},
        "masking_mode": "attractive",
    }
    with tempfile.TemporaryDirectory() as td:
        s1_path  = os.path.join(td, "BNZ_A_1.meta.json")
        s15_path = os.path.join(td, "BNZ_A_1.random.json")
        with open(s1_path, "w") as f:
            json.dump(stage1_data, f)
        result = generate_random_mask_json(s1_path, s15_path, seed=42)
        assert len(result["masked_atom_indices"]) == 3, "Wrong number of random indices"
        mol = Chem.MolFromSmiles(stage1_data["smiles"])
        n_atoms = mol.GetNumAtoms()
        for idx in result["masked_atom_indices"]:
            assert 0 <= idx < n_atoms, f"Index {idx} out of range"
        assert len(set(result["masked_atom_indices"])) == 3, "Duplicate indices found"
    print("✅ Stage 1.5 unit test passed.")
"""

import json
import os
import glob
import random
from typing import Optional

from rdkit import Chem

import config


# ════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def generate_random_mask_json(
    stage1_json_path: str,
    output_path: str,
    seed: int = config.RANDOM_MASK_SEED,
) -> dict:
    """
    Read one Stage-1 .meta.json, sample N random atom indices, write a
    matching random-mask JSON to output_path.

    Parameters
    ----------
    stage1_json_path : str
        Path to a .meta.json produced by stage1_mask_calculation.py.
    output_path : str
        Destination path for the random-mask JSON.
    seed : int
        RNG seed for reproducible sampling (default: config.RANDOM_MASK_SEED).

    Returns
    -------
    dict
        The random-mask metadata dict (same object written to output_path).

    Raises
    ------
    ValueError
        If the Stage-1 JSON has no masked_atom_indices (see Assumption A5).
    ValueError
        If the SMILES in the JSON is not parseable by RDKit.
    """
    with open(stage1_json_path, encoding="utf-8") as f:
        stage1 = json.load(f)

    smiles     = stage1["smiles"]
    s1_indices = stage1.get("masked_atom_indices", [])
    ligand     = stage1.get("ligand", {})

    # ── Validate ──────────────────────────────────────────────────────────────
    if not s1_indices:
        raise ValueError(
            f"Stage-1 JSON has no masked_atom_indices: {stage1_json_path}\n"
            "Skipping this ligand (Assumption A5)."
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES '{smiles}' in {stage1_json_path}")

    n_atoms = mol.GetNumAtoms()
    N       = len(s1_indices)          # target number of random indices

    if N > n_atoms:
        print(
            f"  ⚠️  Ligand {ligand}: N={N} > n_atoms={n_atoms} (Assumption A2 cap applied). "
            f"Using {n_atoms} random indices instead."
        )
        N = n_atoms

    # ── Sample N unique random indices ────────────────────────────────────────
    rng             = random.Random(seed)
    random_indices  = sorted(rng.sample(range(n_atoms), N))

    # ── Build output dict (same format as Stage 1) ────────────────────────────
    random_meta = {
        "smiles":              smiles,
        "masked_atom_indices": random_indices,
        "ligand":              ligand,
        "masking_mode":        "random",
        "source":              "stage1_5",
        "n_atoms":             n_atoms,
        "n_requested":         len(s1_indices),
        "n_applied":           N,
        "seed":                seed,
        # carry over these fields so Stage 2 can use them if needed
        "include_types":       stage1.get("include_types", []),
        "pdb":                 stage1.get("pdb", ""),
        "plip_xml":            stage1.get("plip_xml", ""),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(random_meta, f, indent=2)

    return random_meta


# ════════════════════════════════════════════════════════════════════════════
#  HELPER: ligand key (used to match Stage 1 ↔ Stage 1.5 in Stage 2)
# ════════════════════════════════════════════════════════════════════════════

def ligand_key(meta: dict) -> str:
    """Return a unique string key for a ligand from its meta dict."""
    lig = meta.get("ligand", {})
    return f"{lig.get('resname','?')}-{lig.get('chain','?')}-{lig.get('resseq','?')}"


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(config.RANDOM_MASK_OUTDIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("STAGE 1.5: RANDOM MASK GENERATION")
    print("=" * 60)

    stage1_paths = glob.glob(os.path.join(config.MASK_CALC_OUTDIR, "*.json"))

    if not stage1_paths:
        print(f"  ❌ No JSON files found in {config.MASK_CALC_OUTDIR}")
        print("     Run stage1_mask_calculation.py first.")
        return

    print(f"  Found {len(stage1_paths)} Stage-1 JSON(s) in {config.MASK_CALC_OUTDIR}")

    success, skipped = 0, 0
    for s1_path in sorted(stage1_paths):
        basename   = os.path.splitext(os.path.basename(s1_path))[0]
        out_path   = os.path.join(config.RANDOM_MASK_OUTDIR, basename + ".random.json")

        try:
            meta = generate_random_mask_json(s1_path, out_path, seed=config.RANDOM_MASK_SEED)
            lig  = ligand_key(meta)
            print(f"  ✅ {lig}  →  {len(meta['masked_atom_indices'])} random indices  →  {out_path}")
            success += 1
        except ValueError as e:
            print(f"  ⚠️  Skipped {os.path.basename(s1_path)}: {e}")
            skipped += 1

    print(f"\n✅ Stage 1.5 complete.  {success} written, {skipped} skipped.")
    print(f"   Random-mask JSONs saved to: {config.RANDOM_MASK_OUTDIR}")
    print("   Run stage2_molecule_generation.py next.")


if __name__ == "__main__":
    main()

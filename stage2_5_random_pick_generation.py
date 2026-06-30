# -*- coding: utf-8 -*-
"""
stage2_5_random_pick_generation.py
===================================
Stage 2.5 of the pipeline — random-pick incremental masking generation.

Identical in purpose and output format to Stage 2, but changes HOW the
mask_count atom indices are selected at each step:

  Stage 2   (existing):
    mask_count = k  →  ia_indices[:k]      (always the FIRST k elements)
    mask_count = k  →  rand_indices[:k]    (always the FIRST k elements)

  Stage 2.5 (this file):
    mask_count = k  →  random.sample(ia_indices, k)    (ANY k from the list)
    mask_count = k  →  random.sample(rand_indices, k)  (ANY k from the list)
    Seeded with (base_seed + mask_count) so results are reproducible.

Output files have the same format as Stage 2:
    config.STAGE25_PRED_DIR/<lig_id>/ia_mask{NNN}.txt
    config.STAGE25_PRED_DIR/<lig_id>/rand_mask{NNN}.txt
    config.STAGE25_PLOT_DIR/<lig_id>_incremental_masking.png   (Plot 1 — absolute)
    config.STAGE25_PLOT_DIR/<lig_id>_token_ratio.png           (Plot 2 — yield ratio)

These outputs are consumed by Stages 3–7 unchanged — just point those stages
at STAGE25_PRED_DIR instead of PRED_DIR.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1.  Reads Stage-1 JSONs from config.MASK_CALC_OUTDIR and Stage-1.5 JSONs
     from config.RANDOM_MASK_OUTDIR (same sources as Stage 2).

A2.  For each (ligand, mask_count) cell, samples exactly mask_count indices
     WITHOUT replacement from the full ia_indices / rand_indices lists.
     Seed = config.RANDOM_MASK_SEED + mask_count (so each step has a
     different but reproducible draw).

A3.  N = min(len(ia_indices), len(rand_indices)) — same as Stage 2.

A4.  Output directory is config.STAGE25_PRED_DIR (separate from Stage 2's
     config.PRED_DIR so both stages can be run independently).

A5.  Molecule generation (ChemBERTa), saving, and plotting are imported
     directly from Stage 2 to avoid code duplication.

A6.  config.INCREMENTAL_NUM_SAMPLES samples are generated per cell,
     same as Stage 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python stage2_5_random_pick_generation.py

Must be run after stage1_mask_calculation.py and stage1_5_random_masking.py.

HOW TO TEST (no GPU required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Answer "yes" to the smoke-test prompt.
  Writes to config.STAGE25_PRED_DIR/test/ (wiped each run).
"""

import json
import os
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import config
from rdkit import Chem

# ── Reuse Stage-2 functions unchanged (Assumption A5) ────────────────────────
from stage2_molecule_generation import (
    generate_smiles,
    generate_smiles_sequential,
    get_and_reset_generation_debug_counters,
    load_chemberta,
    load_json_folder,
    mask_atoms_in_smiles,
    plot_incremental_results,
    print_mask_count_generation_debug,
    reset_generation_debug_counters,
    safe_filename,
)
from stage2_token_ratio_plot import plot_token_ratio_results


# ════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PROMPTS
# ════════════════════════════════════════════════════════════════════════════

def _ask_yes_no_default(question: str, default: bool) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"  {question} {hint}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("yes", "y"):
            return True
        if raw in ("no", "n"):
            return False
        print("    Please type 'yes'/'y' or 'no'/'n', or press Enter for the default.")


def _ask_yes_no(question: str) -> bool:
    """Ask without a default — user must explicitly type yes or no."""
    while True:
        raw = input(f"  {question} (yes/no): ").strip().lower()
        if raw in ("yes", "y"):
            return True
        if raw in ("no", "n"):
            return False
        print("    Please type 'yes' or 'no'.")


# ════════════════════════════════════════════════════════════════════════════
#  CORE: RANDOM-PICK INCREMENTAL LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_random_pick_for_ligand(
    lig_id: str,
    smiles: str,
    ia_indices: List[int],
    rand_indices: List[int],
    tokenizer,
    model,
    device: str,
    pred_dir: str,
    num_samples: int = config.INCREMENTAL_NUM_SAMPLES,
    base_seed: int   = config.RANDOM_MASK_SEED,
) -> Tuple[List[int], List[int], List[int], List[float], List[float]]:
    """
    Run the random-pick incremental masking loop for a single ligand.

    For mask_count in 1 … N:
      - Sample mask_count indices from ia_indices   using seed (base_seed + mask_count)
      - Sample mask_count indices from rand_indices using seed (base_seed + mask_count)
      - Generate SMILES from each masked SMILES, record unique valid count

    The seed increments with mask_count so each step picks a genuinely
    different (but reproducible) subset of indices.

    Returns
    -------
    mask_counts    : [1, 2, …, N]
    ia_valids      : unique valid SMILES count per step (IA strategy)
    rand_valids    : unique valid SMILES count per step (random strategy)
    ia_token_pcts  : % of full-SMILES BPE tokens masked (IA), per step
    rand_token_pcts: % of full-SMILES BPE tokens masked (random), per step
    """
    N = min(len(ia_indices), len(rand_indices))
    if len(ia_indices) != len(rand_indices):
        print(
            f"  ⚠️  {lig_id}: IA list length ({len(ia_indices)}) ≠ "
            f"random list length ({len(rand_indices)}). Using N={N}."
        )

    mask_counts: List[int] = []
    ia_valids:   List[int] = []
    rand_valids: List[int] = []
    ia_tok_pcts:   List[float] = []
    rand_tok_pcts: List[float] = []

    lig_pred_dir = os.path.join(pred_dir, safe_filename(lig_id))
    os.makedirs(lig_pred_dir, exist_ok=True)

    mask_token = tokenizer.mask_token
    try:
        _clean_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        total_tokens  = len(
            tokenizer(_clean_smiles, add_special_tokens=False)["input_ids"]
        )
    except Exception:
        total_tokens = 0

    debug = bool(getattr(config, "GENERATION_COUNT_DEBUG", False))

    for mask_count in range(1, N + 1):
        # ── Random-pick: sample mask_count indices without replacement ─────────
        step_seed = base_seed + mask_count

        rng = random.Random(step_seed)
        ia_picked   = rng.sample(ia_indices,   mask_count)

        rng = random.Random(step_seed)
        rand_picked = rng.sample(rand_indices, mask_count)

        print(
            f"    mask_count = {mask_count:>3d}/{N}  "
            f"seed={step_seed}  "
            f"IA picked={sorted(ia_picked)}  "
            f"rand picked={sorted(rand_picked)}",
            end="  ",
            flush=True,
        )

        # ── Interaction-aware masked SMILES ───────────────────────────────────
        ia_masked = mask_atoms_in_smiles(smiles, ia_picked, tokenizer)
        ia_save   = os.path.join(lig_pred_dir, f"ia_mask{mask_count:03d}.txt")
        if debug:
            reset_generation_debug_counters()
        ia_preds  = generate_smiles(
            smiles_masked = ia_masked,
            tokenizer     = tokenizer,
            model         = model,
            device        = device,
            num_samples   = num_samples,
            save_path     = ia_save,
        )
        ia_dbg = get_and_reset_generation_debug_counters() if debug else {}
        n_ia = len(ia_preds)

        # ── Random masked SMILES ──────────────────────────────────────────────
        rand_masked = mask_atoms_in_smiles(smiles, rand_picked, tokenizer)
        rand_save   = os.path.join(lig_pred_dir, f"rand_mask{mask_count:03d}.txt")
        if debug:
            reset_generation_debug_counters()
        rand_preds  = generate_smiles(
            smiles_masked = rand_masked,
            tokenizer     = tokenizer,
            model         = model,
            device        = device,
            num_samples   = num_samples,
            save_path     = rand_save,
        )
        rand_dbg = get_and_reset_generation_debug_counters() if debug else {}
        n_rand = len(rand_preds)

        print(f"IA valid = {n_ia:>4d}   rand valid = {n_rand:>4d}")

        if debug:
            print_mask_count_generation_debug(
                mask_count,
                ia_counts   = ia_dbg,
                rand_counts = rand_dbg,
                num_samples = num_samples,
                fresh_cells = 2,
                cached_cells = 0,
                stage_label = "Stage 2.5",
            )

        mask_counts.append(mask_count)
        ia_valids.append(n_ia)
        rand_valids.append(n_rand)
        if total_tokens > 0:
            ia_tok_pcts.append(100.0 * ia_masked.count(mask_token) / total_tokens)
            rand_tok_pcts.append(100.0 * rand_masked.count(mask_token) / total_tokens)
        else:
            ia_tok_pcts.append(0.0)
            rand_tok_pcts.append(0.0)

    return mask_counts, ia_valids, rand_valids, ia_tok_pcts, rand_tok_pcts


# ════════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_stage25(
    stage1_dir:  Optional[str] = None,
    stage15_dir: Optional[str] = None,
    pred_dir:    Optional[str] = None,
    plot_dir:    Optional[str] = None,
    num_samples: int = config.INCREMENTAL_NUM_SAMPLES,
    base_seed:   int = config.RANDOM_MASK_SEED,
) -> Dict[str, dict]:
    """
    Run Stage 2.5 for all matched ligands.

    Returns
    -------
    results : dict keyed by ligand_id, each value:
        {
          "mask_counts":  [1, 2, …, N],
          "ia_valids":    [int, …],
          "rand_valids":  [int, …],
          "plot_path":    str,
        }
    """
    stage1_dir  = stage1_dir  or config.MASK_CALC_OUTDIR
    stage15_dir = stage15_dir or config.RANDOM_MASK_OUTDIR
    pred_dir    = pred_dir    or config.STAGE25_PRED_DIR
    plot_dir    = plot_dir    or config.STAGE25_PLOT_DIR

    for d in [pred_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    stage1_metas  = load_json_folder(stage1_dir)
    stage15_metas = load_json_folder(stage15_dir)

    print(f"\n  Stage-1   JSONs loaded:  {len(stage1_metas)}")
    print(f"  Stage-1.5 JSONs loaded:  {len(stage15_metas)}")

    matched       = sorted(set(stage1_metas) & set(stage15_metas))
    unmatched_s1  = sorted(set(stage1_metas)  - set(stage15_metas))
    unmatched_s15 = sorted(set(stage15_metas) - set(stage1_metas))

    if unmatched_s1:
        print(f"\n  ⚠️  Stage-1 ligands with NO Stage-1.5 match (skipped): {unmatched_s1}")
    if unmatched_s15:
        print(f"\n  ⚠️  Stage-1.5 ligands with NO Stage-1 match (ignored): {unmatched_s15}")

    if not matched:
        print("\n  ❌ No matched ligands found.  Exiting.")
        return {}

    print(f"\n  ℹ️  Sampling mode  : random pick (NOT prefix slice)")
    print(f"  ℹ️  Base seed      : {base_seed}  (step seed = base + mask_count)")
    print(f"  ℹ️  Output PRED_DIR: {pred_dir}")

    tokenizer, model, device = load_chemberta()

    all_results: Dict[str, dict] = {}

    for lig_id in matched:
        s1  = stage1_metas[lig_id]
        s15 = stage15_metas[lig_id]

        smiles       = s1["smiles"]
        ia_indices   = s1["masked_atom_indices"]
        rand_indices = s15["masked_atom_indices"]

        if not ia_indices:
            print(f"\n  ⚠️  {lig_id}: empty ia_indices, skipping.")
            continue

        N = min(len(ia_indices), len(rand_indices))
        print(f"\n{'='*60}")
        print(f"Ligand : {lig_id}")
        print(f"SMILES : {smiles}")
        print(f"N      : {N}")
        print(f"IA indices (full list)  : {ia_indices}")
        print(f"Rand indices (full list): {rand_indices}")
        print(f"{'='*60}")

        (mask_counts, ia_valids, rand_valids,
         ia_tok_pcts, rand_tok_pcts) = run_random_pick_for_ligand(
            lig_id       = lig_id,
            smiles       = smiles,
            ia_indices   = ia_indices,
            rand_indices = rand_indices,
            tokenizer    = tokenizer,
            model        = model,
            device       = device,
            pred_dir     = pred_dir,
            num_samples  = num_samples,
            base_seed    = base_seed,
        )

        # Plot 1 — absolute: number of masks vs unique valid SMILES.
        plot_path = plot_incremental_results(
            lig_id      = lig_id,
            mask_counts = mask_counts,
            ia_valids   = ia_valids,
            rand_valids = rand_valids,
            plot_dir    = plot_dir,
            num_samples = num_samples,
        )

        # Plot 2 — ratio: % tokens masked vs (unique valid / num_samples).
        ratio_plot_path = plot_token_ratio_results(
            lig_id          = lig_id,
            ia_token_pcts   = ia_tok_pcts,
            rand_token_pcts = rand_tok_pcts,
            ia_valids       = ia_valids,
            rand_valids     = rand_valids,
            plot_dir        = plot_dir,
            total_samples   = num_samples,
            denom_label     = f"num_samples = {num_samples}",
        )

        all_results[lig_id] = {
            "mask_counts":     mask_counts,
            "ia_valids":       ia_valids,
            "rand_valids":     rand_valids,
            "ia_token_pcts":   ia_tok_pcts,
            "rand_token_pcts": rand_tok_pcts,
            "plot_path":       plot_path,
            "ratio_plot_path": ratio_plot_path,
        }

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED TEST  (no GPU required)
# ════════════════════════════════════════════════════════════════════════════

def _run_test() -> bool:
    """
    Smoke test — verifies random-pick index selection logic without
    loading ChemBERTa or writing real predictions.
    """
    print("\n" + "=" * 60)
    print("STAGE 2.5 SELF-TEST  (index-selection logic only)")
    print("=" * 60)

    td = os.path.join(config.STAGE25_PRED_DIR, "test")
    if os.path.exists(td):
        shutil.rmtree(td)
        print(f"  🗑  Wiped existing test dir: {td}")
    os.makedirs(td)
    print(f"  📁 Created fresh test dir:  {td}\n")

    passed = 0
    failed = 0

    def check(cond: bool, msg: str):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ PASS  {msg}")
            passed += 1
        else:
            print(f"  ❌ FAIL  {msg}")
            failed += 1

    ia_indices   = [0, 3, 7, 11, 15]
    rand_indices = [2, 5, 9, 12, 14]
    base_seed    = config.RANDOM_MASK_SEED
    N            = min(len(ia_indices), len(rand_indices))

    all_ia_picks:   Dict[int, List[int]] = {}
    all_rand_picks: Dict[int, List[int]] = {}

    for mask_count in range(1, N + 1):
        step_seed = base_seed + mask_count

        rng = random.Random(step_seed)
        ia_picked   = rng.sample(ia_indices, mask_count)

        rng = random.Random(step_seed)
        rand_picked = rng.sample(rand_indices, mask_count)

        all_ia_picks[mask_count]   = ia_picked
        all_rand_picks[mask_count] = rand_picked

        # ── Correctness checks per step ───────────────────────────────────────
        check(len(ia_picked)   == mask_count,
              f"mask_count={mask_count}: IA pick has correct length")
        check(len(rand_picked) == mask_count,
              f"mask_count={mask_count}: rand pick has correct length")
        check(all(idx in ia_indices   for idx in ia_picked),
              f"mask_count={mask_count}: all IA picks are from ia_indices")
        check(all(idx in rand_indices for idx in rand_picked),
              f"mask_count={mask_count}: all rand picks are from rand_indices")
        check(len(set(ia_picked))   == mask_count,
              f"mask_count={mask_count}: no duplicate IA picks")
        check(len(set(rand_picked)) == mask_count,
              f"mask_count={mask_count}: no duplicate rand picks")

    # ── Reproducibility: re-running with same seeds gives same picks ──────────
    for mask_count in range(1, N + 1):
        step_seed = base_seed + mask_count
        rng = random.Random(step_seed)
        ia_pick2 = rng.sample(ia_indices, mask_count)
        check(ia_pick2 == all_ia_picks[mask_count],
              f"mask_count={mask_count}: picks are reproducible with same seed")

    # ── Stage 2 comparison: verify picks differ from prefix slice ─────────────
    # (They may accidentally match for mask_count=1 but should diverge overall)
    differs = sum(
        1 for mc in range(1, N + 1)
        if sorted(all_ia_picks[mc]) != sorted(ia_indices[:mc])
    )
    check(differs > 0,
          f"random-pick selections differ from Stage-2 prefix at ≥1 step "
          f"(differed at {differs}/{N} steps)")

    print(f"\n  Seed behaviour:")
    for mc in range(1, N + 1):
        print(f"    mask_count={mc}  seed={base_seed+mc}"
              f"  IA→{sorted(all_ia_picks[mc])}"
              f"  rand→{sorted(all_rand_picks[mc])}")

    print(f"\n{'='*60}")
    print(f"Test complete:  {passed} passed,  {failed} failed.")
    print("✅ All tests passed." if failed == 0 else "❌ Some tests failed.")
    print(f"\n  Test outputs preserved at: {td}")
    return failed == 0


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("STAGE 2.5: RANDOM-PICK INCREMENTAL MASKING GENERATION")
    print("=" * 60)

    print(f"""
  This stage is identical to Stage 2 except for HOW atom indices are
  selected at each mask_count step:

    Stage 2   : ia_indices[:k]           — always the FIRST k elements
    Stage 2.5 : random.sample(ia_indices, k)  — ANY k elements (seeded)

  Seed per step = config.RANDOM_MASK_SEED + mask_count = {config.RANDOM_MASK_SEED} + mask_count

  Inputs  (same as Stage 2):
    Stage-1   JSONs : {config.MASK_CALC_OUTDIR}
    Stage-1.5 JSONs : {config.RANDOM_MASK_OUTDIR}

  Outputs (separate from Stage 2):
    Predictions : {config.STAGE25_PRED_DIR}
    Plots       : {config.STAGE25_PLOT_DIR}

  Output files are format-compatible with Stages 3–7.
  To use them in Stage 3, run:
      python stage3_analysis.py
  and when prompted for the pred_dir, use:
      {config.STAGE25_PRED_DIR}

  Smoke test verifies index-selection logic — no GPU needed.
""")

    run_test = _ask_yes_no_default("Run the smoke test?", default=False)
    if run_test:
        ok = _run_test()
        sys.exit(0 if ok else 1)

    run_full = _ask_yes_no("Run the full generation? (requires GPU + ChemBERTa)")
    if not run_full:
        print("  Exiting.")
        sys.exit(0)

    results = run_stage25()

    print("\n" + "=" * 60)
    print("✅ Stage 2.5 complete.")
    print(f"   Predictions : {config.STAGE25_PRED_DIR}")
    print(f"   Plots       : {config.STAGE25_PLOT_DIR}")
    print(f"   Ligands     : {len(results)}")
    for lig_id, r in sorted(results.items()):
        print(f"   • {lig_id}  N={len(r['mask_counts'])}")


if __name__ == "__main__":
    main()

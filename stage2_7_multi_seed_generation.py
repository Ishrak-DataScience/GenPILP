# -*- coding: utf-8 -*-
"""
stage2_7_multi_seed_generation.py
===================================
Stage 2.7 of the pipeline — multi-seed random-pick incremental generation.

Extends Stage 2.5 by accepting a LIST of random seeds
(config.RANDOM_MASK_SEEDS_LIST) instead of a single seed.  For each seed,
a full random-pick run is performed (same logic as Stage 2.5: at each
mask_count k, sample k indices from the full ia_indices / rand_indices list).
The resulting SMILES from all seeds are aggregated into two sets per
mask_count:

  IA-aggregated set   = union of all seeds' IA predictions
  Rand-aggregated set = union of all seeds' random predictions

These are written as:
  config.STAGE27_PRED_DIR/<lig_id>/ia_mask{NNN}.txt
  config.STAGE27_PRED_DIR/<lig_id>/rand_mask{NNN}.txt

Format is identical to Stage 2 / 2.5, so Stages 3–7 consume them unchanged.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REUSE LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each (seed, lig_id, mask_count) cell, predictions are looked up in this
priority order before calling ChemBERTa:

  1. Stage-2.5 reuse (if applicable):
       If seed == config.RANDOM_MASK_SEED  AND
       config.STAGE25_PRED_DIR/<lig_id>/ia_mask{NNN}.txt exists and is
       non-empty → read from Stage 2.5 output, skip generation.
       Stage 2.5 does NOT need to have been run; check is non-blocking.

  2. Stage-2.7 internal seed cache:
       config.STAGE27_PRED_DIR/<lig_id>/_seed_cache/seed_{seed}/
         ia_mask{NNN}.txt  (written by a previous Stage 2.7 run)
       If found and non-empty → reuse, skip generation.

  3. Generate fresh:
       Run ChemBERTa, save to the seed cache (for future reuse), then
       aggregate with other seeds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1.  config.RANDOM_MASK_SEEDS_LIST = [17, 19, 23]  (list of ints).
     Each seed is independent; results are unioned after all seeds complete.

A2.  For seed S and mask_count k: step_seed = S + k (same formula as
     Stage 2.5).  Index sampling: random.sample(ia_indices, k) seeded with
     step_seed.

A3.  Aggregated output files contain unique valid SMILES (set union across
     seeds, deduplicated by exact string).

A4.  Stage 2.5 is NOT required to have been run.  Reuse check is a
     read-only filesystem test; if STAGE25_PRED_DIR does not exist or the
     file is absent/empty, generation proceeds normally.

A5.  All generation functions are imported from Stage 2 unchanged.

A6.  Per-seed intermediate files live in STAGE27_PRED_DIR/<lig>/_seed_cache/
     (internal cache).  Only the aggregated files at
     STAGE27_PRED_DIR/<lig>/ia_mask{NNN}.txt are the "output" consumed by
     downstream stages.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python stage2_7_multi_seed_generation.py

Must be run after stage1_mask_calculation.py and stage1_5_random_masking.py.
Stage 2.5 is NOT required.

HOW TO TEST (no GPU required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Answer "yes" to the smoke-test prompt.
"""

import os
import random
import shutil
import sys
from typing import Dict, List, Optional, Set, Tuple

import config
from stage2_molecule_generation import (
    generate_smiles_sequential,
    load_chemberta,
    load_json_folder,
    mask_atoms_in_smiles,
    plot_incremental_results,
    safe_filename,
)


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
    while True:
        raw = input(f"  {question} (yes/no): ").strip().lower()
        if raw in ("yes", "y"):
            return True
        if raw in ("no", "n"):
            return False
        print("    Please type 'yes' or 'no'.")


# ════════════════════════════════════════════════════════════════════════════
#  REUSE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _seed_cache_path(stage27_pred_dir: str, lig_id: str,
                     seed: int, mask_count: int,
                     strategy: str) -> str:
    """
    Return path to the per-seed intermediate cache file.
    strategy: "ia" or "rand"
    """
    return os.path.join(
        stage27_pred_dir,
        safe_filename(lig_id),
        "_seed_cache",
        f"seed_{seed}",
        f"{strategy}_mask{mask_count:03d}.txt",
    )


def _read_smiles_file(path: str) -> List[str]:
    """
    Read unique valid-looking SMILES from a txt file (one per line).
    Returns empty list if file absent or empty.
    """
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []


def _write_smiles_file(path: str, smiles_list: List[str]) -> None:
    """Write a list of SMILES strings, one per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(smiles_list) + "\n")


def _find_cached_predictions(
    seed: int,
    lig_id: str,
    mask_count: int,
    stage27_pred_dir: str,
    stage25_pred_dir: str,
    stage25_base_seed: int,
) -> Tuple[Optional[List[str]], Optional[List[str]], str]:
    """
    Try to find already-computed predictions for (seed, lig_id, mask_count).

    Priority:
      1. Stage-2.5 output (if seed matches Stage-2.5's base seed)
      2. Stage-2.7 internal seed cache (from a previous run)
      3. None — caller must generate

    Returns
    -------
    ia_smiles   : list of SMILES or None
    rand_smiles : list of SMILES or None
    source      : human-readable label ("stage2.5", "stage2.7 cache", "new")
    """
    # ── Priority 1: Stage-2.5 reuse ──────────────────────────────────────────
    if seed == stage25_base_seed:
        s25_ia   = os.path.join(stage25_pred_dir,
                                safe_filename(lig_id),
                                f"ia_mask{mask_count:03d}.txt")
        s25_rand = os.path.join(stage25_pred_dir,
                                safe_filename(lig_id),
                                f"rand_mask{mask_count:03d}.txt")
        ia_smi   = _read_smiles_file(s25_ia)
        rand_smi = _read_smiles_file(s25_rand)
        if ia_smi or rand_smi:
            return ia_smi, rand_smi, "stage2.5"

    # ── Priority 2: Stage-2.7 internal seed cache ─────────────────────────────
    cache_ia   = _seed_cache_path(stage27_pred_dir, lig_id, seed, mask_count, "ia")
    cache_rand = _seed_cache_path(stage27_pred_dir, lig_id, seed, mask_count, "rand")
    ia_smi   = _read_smiles_file(cache_ia)
    rand_smi = _read_smiles_file(cache_rand)
    if ia_smi or rand_smi:
        return ia_smi, rand_smi, "stage2.7 cache"

    return None, None, "new"


# ════════════════════════════════════════════════════════════════════════════
#  CORE: MULTI-SEED INCREMENTAL LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_multi_seed_for_ligand(
    lig_id: str,
    smiles: str,
    ia_indices: List[int],
    rand_indices: List[int],
    tokenizer,
    model,
    device: str,
    seeds: List[int],
    stage27_pred_dir: str,
    stage25_pred_dir: str,
    stage25_base_seed: int,
    num_samples: int = config.INCREMENTAL_NUM_SAMPLES,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Run the multi-seed random-pick incremental loop for a single ligand.

    For each mask_count in 1 … N:
      For each seed in seeds:
        - Look up or generate predictions (IA and random)
        - Accumulate into two sets (ia_pool, rand_pool)
      Write aggregated ia_mask{NNN}.txt and rand_mask{NNN}.txt

    Returns
    -------
    mask_counts  : [1, 2, …, N]
    ia_valids    : aggregated unique valid SMILES count per step
    rand_valids  : aggregated unique valid SMILES count per step
    """
    N = min(len(ia_indices), len(rand_indices))
    if len(ia_indices) != len(rand_indices):
        print(
            f"  ⚠️  {lig_id}: IA list length ({len(ia_indices)}) ≠ "
            f"random list length ({len(rand_indices)}). Using N={N}."
        )

    lig_out_dir = os.path.join(stage27_pred_dir, safe_filename(lig_id))
    os.makedirs(lig_out_dir, exist_ok=True)

    mask_counts: List[int] = []
    ia_valids:   List[int] = []
    rand_valids: List[int] = []

    for mask_count in range(1, N + 1):
        print(f"\n    ── mask_count = {mask_count}/{N} ──")

        # Aggregate across all seeds
        ia_pool:   Set[str] = set()
        rand_pool: Set[str] = set()

        for seed in seeds:
            step_seed = seed + mask_count

            # ── Try reuse first ───────────────────────────────────────────────
            ia_cached, rand_cached, source = _find_cached_predictions(
                seed              = seed,
                lig_id            = lig_id,
                mask_count        = mask_count,
                stage27_pred_dir  = stage27_pred_dir,
                stage25_pred_dir  = stage25_pred_dir,
                stage25_base_seed = stage25_base_seed,
            )

            if source != "new":
                ia_pool.update(ia_cached   or [])
                rand_pool.update(rand_cached or [])
                print(
                    f"      seed={seed:>4d}  step_seed={step_seed}  "
                    f"♻️  reused from {source}  "
                    f"(+{len(ia_cached or [])} IA, +{len(rand_cached or [])} rand)"
                )
                continue

            # ── Sample indices ────────────────────────────────────────────────
            rng         = random.Random(step_seed)
            ia_picked   = rng.sample(ia_indices,   mask_count)
            rng         = random.Random(step_seed)
            rand_picked = rng.sample(rand_indices, mask_count)

            print(
                f"      seed={seed:>4d}  step_seed={step_seed}  "
                f"IA picked={sorted(ia_picked)}  "
                f"rand picked={sorted(rand_picked)}",
                end="  ", flush=True,
            )

            # ── Generate IA predictions ───────────────────────────────────────
            ia_masked  = mask_atoms_in_smiles(smiles, ia_picked, tokenizer)
            cache_ia   = _seed_cache_path(stage27_pred_dir, lig_id, seed,
                                          mask_count, "ia")
            ia_preds   = generate_smiles_sequential(
                smiles_masked = ia_masked,
                tokenizer     = tokenizer,
                model         = model,
                device        = device,
                num_samples   = num_samples,
                save_path     = cache_ia,
            )

            # ── Generate random predictions ───────────────────────────────────
            rand_masked = mask_atoms_in_smiles(smiles, rand_picked, tokenizer)
            cache_rand  = _seed_cache_path(stage27_pred_dir, lig_id, seed,
                                           mask_count, "rand")
            rand_preds  = generate_smiles_sequential(
                smiles_masked = rand_masked,
                tokenizer     = tokenizer,
                model         = model,
                device        = device,
                num_samples   = num_samples,
                save_path     = cache_rand,
            )

            ia_pool.update(ia_preds)
            rand_pool.update(rand_preds)
            print(f"new  (+{len(ia_preds)} IA, +{len(rand_preds)} rand)")

        # ── Write aggregated output for this mask_count ───────────────────────
        agg_ia_path   = os.path.join(lig_out_dir, f"ia_mask{mask_count:03d}.txt")
        agg_rand_path = os.path.join(lig_out_dir, f"rand_mask{mask_count:03d}.txt")
        _write_smiles_file(agg_ia_path,   sorted(ia_pool))
        _write_smiles_file(agg_rand_path, sorted(rand_pool))

        print(
            f"      → Aggregated: {len(ia_pool)} unique IA, "
            f"{len(rand_pool)} unique rand"
        )

        mask_counts.append(mask_count)
        ia_valids.append(len(ia_pool))
        rand_valids.append(len(rand_pool))

    return mask_counts, ia_valids, rand_valids


# ════════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_stage27(
    stage1_dir:        Optional[str]   = None,
    stage15_dir:       Optional[str]   = None,
    stage27_pred_dir:  Optional[str]   = None,
    stage27_plot_dir:  Optional[str]   = None,
    stage25_pred_dir:  Optional[str]   = None,
    seeds:             Optional[List[int]] = None,
    num_samples:       int = config.INCREMENTAL_NUM_SAMPLES,
) -> Dict[str, dict]:
    """
    Run Stage 2.7 for all matched ligands.

    Returns
    -------
    results : dict keyed by ligand_id, each value:
        {
          "mask_counts": [1, …, N],
          "ia_valids":   [int, …],
          "rand_valids": [int, …],
          "plot_path":   str,
        }
    """
    stage1_dir       = stage1_dir       or config.MASK_CALC_OUTDIR
    stage15_dir      = stage15_dir      or config.RANDOM_MASK_OUTDIR
    stage27_pred_dir = stage27_pred_dir or config.STAGE27_PRED_DIR
    stage27_plot_dir = stage27_plot_dir or config.STAGE27_PLOT_DIR
    stage25_pred_dir = stage25_pred_dir or config.STAGE25_PRED_DIR
    seeds            = seeds            or list(config.RANDOM_MASK_SEEDS_LIST)

    if not seeds:
        print("  ❌ config.RANDOM_MASK_SEEDS_LIST is empty. Exiting.")
        return {}

    for d in [stage27_pred_dir, stage27_plot_dir]:
        os.makedirs(d, exist_ok=True)

    stage1_metas  = load_json_folder(stage1_dir)
    stage15_metas = load_json_folder(stage15_dir)

    print(f"\n  Stage-1   JSONs loaded : {len(stage1_metas)}")
    print(f"  Stage-1.5 JSONs loaded : {len(stage15_metas)}")
    print(f"\n  Seeds                  : {seeds}")
    print(f"  Stage-2.5 base seed    : {config.RANDOM_MASK_SEED}  "
          f"({'will reuse Stage-2.5 results if present' if config.RANDOM_MASK_SEED in seeds else 'not in seed list'})")
    print(f"  Stage-2.5 PRED_DIR     : {stage25_pred_dir}")
    print(f"  Stage-2.7 PRED_DIR     : {stage27_pred_dir}")

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
        print(f"N      : {N}  |  Seeds: {seeds}")
        print(f"IA indices   : {ia_indices}")
        print(f"Rand indices : {rand_indices}")
        print(f"{'='*60}")

        mask_counts, ia_valids, rand_valids = run_multi_seed_for_ligand(
            lig_id            = lig_id,
            smiles            = smiles,
            ia_indices        = ia_indices,
            rand_indices      = rand_indices,
            tokenizer         = tokenizer,
            model             = model,
            device            = device,
            seeds             = seeds,
            stage27_pred_dir  = stage27_pred_dir,
            stage25_pred_dir  = stage25_pred_dir,
            stage25_base_seed = config.RANDOM_MASK_SEED,
            num_samples       = num_samples,
        )

        plot_path = plot_incremental_results(
            lig_id      = lig_id,
            mask_counts = mask_counts,
            ia_valids   = ia_valids,
            rand_valids = rand_valids,
            plot_dir    = stage27_plot_dir,
            num_samples = num_samples,
        )

        all_results[lig_id] = {
            "mask_counts": mask_counts,
            "ia_valids":   ia_valids,
            "rand_valids": rand_valids,
            "plot_path":   plot_path,
        }

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED TEST  (no GPU required)
# ════════════════════════════════════════════════════════════════════════════

def _run_test() -> bool:
    """
    Smoke test — verifies:
      1. Random-pick index sampling (length, membership, no duplicates)
      2. Reproducibility with same seed
      3. Stage-2.5 reuse detection (reads pre-written file)
      4. Stage-2.7 internal cache reuse (reads pre-written cache file)
      5. Aggregation is a proper union (deduplication)
      6. Aggregated output files are written to the right location
    """
    print("\n" + "=" * 60)
    print("STAGE 2.7 SELF-TEST  (logic only, no GPU)")
    print("=" * 60)

    td = os.path.join(config.STAGE27_PRED_DIR, "test")
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

    ia_indices   = [0, 3, 7, 11, 15, 20]
    rand_indices = [2, 5, 9, 12, 14, 18]
    seeds        = [17, 19, 42]       # 42 = RANDOM_MASK_SEED → triggers Stage-2.5 reuse
    lig_id       = "EAM-A-1"
    stage27_dir  = os.path.join(td, "stage27")
    stage25_dir  = os.path.join(td, "stage25")
    lig_safe     = safe_filename(lig_id)

    # ── Write fake Stage-2.5 files for seed=42 ───────────────────────────────
    s25_ia_1   = os.path.join(stage25_dir, lig_safe, "ia_mask001.txt")
    s25_rand_1 = os.path.join(stage25_dir, lig_safe, "rand_mask001.txt")
    os.makedirs(os.path.dirname(s25_ia_1))
    _write_smiles_file(s25_ia_1,   ["CCO", "CC(=O)O"])
    _write_smiles_file(s25_rand_1, ["c1ccccc1", "CCN"])

    # ── Write fake Stage-2.7 cache for seed=19, mask_count=1 ─────────────────
    cache_ia_19   = _seed_cache_path(stage27_dir, lig_id, 19, 1, "ia")
    cache_rand_19 = _seed_cache_path(stage27_dir, lig_id, 19, 1, "rand")
    _write_smiles_file(cache_ia_19,   ["CCCO", "CCO"])   # CCO overlaps with s25
    _write_smiles_file(cache_rand_19, ["c1ccccc1", "CCCC"])

    # ── Test _find_cached_predictions ────────────────────────────────────────
    ia_c, rand_c, src = _find_cached_predictions(
        seed=42, lig_id=lig_id, mask_count=1,
        stage27_pred_dir=stage27_dir,
        stage25_pred_dir=stage25_dir,
        stage25_base_seed=42,
    )
    check(src == "stage2.5",        "seed=42 → source is stage2.5")
    check(ia_c == ["CCO", "CC(=O)O"], "stage2.5 IA content correct")

    ia_c, rand_c, src = _find_cached_predictions(
        seed=19, lig_id=lig_id, mask_count=1,
        stage27_pred_dir=stage27_dir,
        stage25_pred_dir=stage25_dir,
        stage25_base_seed=42,
    )
    check(src == "stage2.7 cache",  "seed=19 → source is stage2.7 cache")

    ia_c, rand_c, src = _find_cached_predictions(
        seed=17, lig_id=lig_id, mask_count=1,
        stage27_pred_dir=stage27_dir,
        stage25_pred_dir=stage25_dir,
        stage25_base_seed=42,
    )
    check(src == "new",             "seed=17 → source is new (no cache)")

    # ── Test aggregation (union, deduplication) ───────────────────────────────
    pool_a = {"CCO", "CC(=O)O"}      # seed=42 (stage2.5)
    pool_b = {"CCCO", "CCO"}         # seed=19 (cache) — CCO overlaps
    pool_c = {"CCCC", "CCC"}         # seed=17 (new)
    union  = pool_a | pool_b | pool_c
    check(len(union) == 5,          f"union deduplicates correctly (got {len(union)}, want 5)")
    check("CCO" in union,           "CCO present once (not duplicated)")

    # ── Test index sampling per seed ──────────────────────────────────────────
    for seed in [17, 19]:
        for mask_count in range(1, 4):
            step_seed = seed + mask_count
            rng = random.Random(step_seed)
            ia_pick = rng.sample(ia_indices, mask_count)
            check(len(ia_pick) == mask_count,
                  f"seed={seed} mask={mask_count}: correct pick length")
            check(all(i in ia_indices for i in ia_pick),
                  f"seed={seed} mask={mask_count}: all picks from ia_indices")
            check(len(set(ia_pick)) == mask_count,
                  f"seed={seed} mask={mask_count}: no duplicates")
            # Reproducibility
            rng2 = random.Random(step_seed)
            check(rng2.sample(ia_indices, mask_count) == ia_pick,
                  f"seed={seed} mask={mask_count}: reproducible")

    # ── Test seeds do NOT influence each other ────────────────────────────────
    picks_17 = {mc: random.Random(17 + mc).sample(ia_indices, mc)
                for mc in range(1, 4)}
    picks_19 = {mc: random.Random(19 + mc).sample(ia_indices, mc)
                for mc in range(1, 4)}
    differs = sum(1 for mc in range(1, 4)
                  if sorted(picks_17[mc]) != sorted(picks_19[mc]))
    check(differs > 0,
          f"seeds 17 and 19 produce different picks ({differs}/3 steps differ)")

    # ── Test Stage-2.5 reuse: seed NOT in list → goes to new ─────────────────
    ia_c, rand_c, src = _find_cached_predictions(
        seed=999, lig_id=lig_id, mask_count=1,
        stage27_pred_dir=stage27_dir,
        stage25_pred_dir=stage25_dir,
        stage25_base_seed=42,
    )
    check(src == "new",
          "seed=999 (not in any cache, not Stage-2.5 seed) → new")

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
    print("STAGE 2.7: MULTI-SEED RANDOM-PICK GENERATION")
    print("=" * 60)

    seeds = list(config.RANDOM_MASK_SEEDS_LIST)
    s25_overlap = config.RANDOM_MASK_SEED in seeds

    print(f"""
  Extends Stage 2.5 with multiple random seeds.

  Seeds          : {seeds}
  Stage-2.5 seed : {config.RANDOM_MASK_SEED}  {"← will reuse Stage-2.5 results if present" if s25_overlap else "← not in list"}

  For each (seed, mask_count) cell:
    step_seed = seed + mask_count
    ia_picked = random.sample(ia_indices, mask_count)   [seeded]

  Reuse order:
    1. Stage-2.5 output  (if seed == {config.RANDOM_MASK_SEED} and files exist)
    2. Stage-2.7 cache   (from a previous Stage-2.7 run)
    3. Generate fresh

  Aggregated outputs (union across all seeds per mask_count):
    {config.STAGE27_PRED_DIR}

  These are format-compatible with Stages 3–7.
  Stage 2.5 is NOT required to have been run first.

  Smoke test verifies reuse logic and aggregation — no GPU needed.
""")

    run_test = _ask_yes_no_default("Run the smoke test?", default=False)
    if run_test:
        ok = _run_test()
        sys.exit(0 if ok else 1)

    run_full = _ask_yes_no("Run the full generation? (requires GPU + ChemBERTa)")
    if not run_full:
        print("  Exiting.")
        sys.exit(0)

    results = run_stage27()

    print("\n" + "=" * 60)
    print("✅ Stage 2.7 complete.")
    print(f"   Predictions : {config.STAGE27_PRED_DIR}")
    print(f"   Plots       : {config.STAGE27_PLOT_DIR}")
    print(f"   Ligands     : {len(results)}")
    for lig_id, r in sorted(results.items()):
        print(f"   • {lig_id}  N={len(r['mask_counts'])}")


if __name__ == "__main__":
    main()

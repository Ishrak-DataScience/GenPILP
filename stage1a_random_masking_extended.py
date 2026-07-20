# -*- coding: utf-8 -*-
"""
stage1a_random_masking_extended.py
===================================
Extended Stage 1a: everything stage1a_random_masking.py does, plus three
optional candle (box-and-whisker) chart sets exploring variance/deviation
that the single-line summary plot collapses away.

    1. Unique-valid-ratio spread ACROSS SEEDS
       One figure per temperature. x = mask %, candle = spread of the
       per-seed "unique valid / total" ratio (only len(seeds) points per
       candle — this directly shows how sensitive the yield metric is to
       which tokens happened to get masked, since seed drives both mask
       position AND the ChemBERTa sampling draw).

    2. Pairwise-Tanimoto-similarity spread PER (seed, temperature)
       One figure per (seed, temperature) combo (3 seeds x 5 temperatures
       = 15 figures by default). x = mask %, candle = distribution of
       pairwise Tanimoto similarity among that combo's valid generated
       molecules.

    3. Pairwise-Tanimoto-similarity spread AGGREGATED ACROSS SEEDS
       One figure per temperature (5 figures by default). Two different,
       both user-selectable, definitions of "aggregated" are supported —
       see DESIGN DECISIONS D2 below.

All three chart sets are optional and asked about interactively at
runtime (matching this project's Stage-3 convention — see
stage3_analysis.ask_runtime_options / _ask_yes_no_default).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN DECISIONS (resolved with the user; see conversation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
D1. "Pairwise similarity" = Tanimoto similarity (Morgan fingerprints,
    radius=2, 2048 bits) among the generated molecules THEMSELVES —
    reusing stage3_analysis.pairwise_tanimoto verbatim — NOT similarity of
    each generated molecule vs its own original/parent SMILES (that's a
    different, already-implemented metric in stage3_analysis.py,
    vs_original_tanimoto, not requested here). This is also the only
    coherent reading given Stage 1a's data model: each combo produces
    exactly ONE generated candidate per input molecule, so "pairwise"
    necessarily means comparing across different input molecules' single
    outputs within the same (percent, temperature, seed) combo.

D2. The "aggregated across seeds" plot set supports BOTH of two
    aggregation definitions, chosen at runtime (or both, producing two
    parallel sets of 5 figures each):
      "pooled" — merge all 3 seeds' valid generated molecules into one
                 pool (deduplicated across the WHOLE pool, if
                 deduplication is on) and compute pairwise_tanimoto ONCE
                 over that pool. Includes cross-seed pairs (e.g. a
                 molecule from seed 17 vs one from seed 53).
      "concat" — compute pairwise_tanimoto separately WITHIN each seed's
                 own generated set (deduplicated per-seed, if on), then
                 concatenate the 3 resulting score lists for display.
                 Never compares molecules across different seeds.

D3. Deduplication before similarity: matches stage3_analysis.py's own
    "Deduplicate pooled SMILES?" convention exactly — an interactive
    yes/no prompt (default: yes), rather than a hardcoded choice, using
    RDKit-canonical SMILES as the identity (same as Stage 1a's own
    unique-valid-ratio metric, D6 in stage1a_random_masking.py).

D4. Candle = a standard box-and-whisker plot (matplotlib boxplot):
    median line, IQR box, whiskers, outlier points. A literal OHLC
    financial candlestick has no natural open/close definition for a
    plain value distribution.

D5. "Use generated data from stage1a_random_masking.py if available,
    else generate it": if any combo CSVs already exist in
    config.STAGE1A_DIR, an interactive prompt asks whether to reuse them
    (default: yes — resume/skip already-done molecules) or wipe them and
    regenerate everything from scratch (a second, default-no confirmation
    guards the actual delete, since it's destructive). Either way,
    run_stage1a() + summarize_and_plot_stage1a() are then called — both
    are resumable/idempotent, so this transparently fills in whatever's
    missing without recomputing what's already there.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage1a_random_masking_extended.py

HOW TO TEST (quick smoke test, no full ChEMBL download needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage1a_random_masking_extended.py --test
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from stage1a_random_masking import (
    _canonical_smiles,
    combo_csv_path,
    compute_combo_metric,
    run_stage1a,
    summarize_and_plot_stage1a,
)
from stage3_analysis import _ask_yes_no_default, pairwise_tanimoto


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_valid_generated_smiles(
    percent: float,
    temperature: float,
    seed: int,
    output_dir: str,
    dedup: bool,
) -> List[str]:
    """
    Read one combo CSV and return its valid generated_smiles values.

    dedup=True canonicalizes each SMILES (RDKit) and drops repeats, so a
    molecule ChemBERTa regenerated identically for two different input
    molecules counts once (same identity rule as Stage 1a's own
    unique-valid-ratio metric).
    """
    csv_path = combo_csv_path(output_dir, percent, temperature, seed)
    if not os.path.isfile(csv_path):
        return []

    out: List[str] = []
    seen: set = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("valid") != "True" or not row.get("generated_smiles"):
                continue
            smi = row["generated_smiles"]
            if not dedup:
                out.append(smi)
                continue
            canon = _canonical_smiles(smi)
            if canon and canon not in seen:
                seen.add(canon)
                out.append(canon)
    return out


# ════════════════════════════════════════════════════════════════════════════
#  CANDLE (BOX-AND-WHISKER) CHART — shared renderer
# ════════════════════════════════════════════════════════════════════════════

def _box_width(positions: List[float]) -> float:
    if len(positions) < 2:
        return 2.0
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    return max(min(gaps) * 0.6, 0.1)


def draw_candle_chart(
    percent_to_values: Dict[float, List[float]],
    title: str,
    ylabel: str,
    plot_path: str,
    color: str = "#2a78d6",  # dataviz palette slot 1 (blue) — single-series chart
) -> Optional[str]:
    """
    Box-and-whisker chart: x = mask percent, one box per percent showing the
    spread of percent_to_values[percent]. Percents with an empty value list
    are skipped (with a warning) rather than drawn as an empty box.
    """
    percents = sorted(p for p, vals in percent_to_values.items() if vals)
    if not percents:
        print(f"  ⚠️  No data for '{title}' — candle chart skipped.")
        return None

    data = [percent_to_values[p] for p in percents]

    surface, gridline, axis_ink, label_ink, title_ink, box_edge = (
        "#fcfcfb", "#e1e0d9", "#898781", "#52514e", "#0b0b0b", "#0b0b0b",
    )

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.boxplot(
        data,
        positions=percents,
        widths=_box_width(percents),
        patch_artist=True,
        boxprops=dict(facecolor=color, edgecolor=box_edge, alpha=0.75),
        medianprops=dict(color=box_edge, linewidth=1.5),
        whiskerprops=dict(color=label_ink),
        capprops=dict(color=label_ink),
        flierprops=dict(markerfacecolor=color, markeredgecolor=label_ink, markersize=4, alpha=0.6),
    )

    ax.set_xlabel("Tokens masked (%)", fontsize=12, color=label_ink)
    ax.set_ylabel(ylabel, fontsize=12, color=label_ink)
    ax.set_title(title, fontsize=13, color=title_ink)
    ax.set_xticks(percents)
    ax.set_xticklabels([f"{p:g}" for p in percents])
    ax.tick_params(colors=axis_ink)
    ax.grid(True, axis="y", color=gridline, linewidth=1)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Candle chart saved: {plot_path}")
    return plot_path


# ════════════════════════════════════════════════════════════════════════════
#  PLOT SET 1: unique-valid ratio spread ACROSS SEEDS (D-none, reuses D6 metric)
# ════════════════════════════════════════════════════════════════════════════

def plot_unique_valid_seed_candles(
    percents: List[float],
    temperatures: List[float],
    seeds: List[int],
    output_dir: str,
) -> List[str]:
    """One candle chart per temperature; candle spread = per-seed ratio."""
    out_subdir = os.path.join(output_dir, "candles_unique_valid_by_seed")
    saved: List[str] = []

    for temperature in temperatures:
        percent_to_values: Dict[float, List[float]] = {}
        for percent in percents:
            values = []
            for seed in seeds:
                csv_path = combo_csv_path(output_dir, percent, temperature, seed)
                if not os.path.isfile(csv_path):
                    continue
                ratio, _total = compute_combo_metric(csv_path)
                values.append(ratio)
            percent_to_values[percent] = values

        path = draw_candle_chart(
            percent_to_values,
            title=(
                f"Unique-valid ratio: spread across seeds (T={temperature:g})\n"
                f"seeds={seeds}"
            ),
            ylabel="Unique valid molecules / total\n(one point per seed)",
            plot_path=os.path.join(out_subdir, f"unique_valid_candles_T{temperature:g}.png"),
        )
        if path:
            saved.append(path)
    return saved


# ════════════════════════════════════════════════════════════════════════════
#  PLOT SET 2: pairwise-Tanimoto spread PER (seed, temperature)  [D1]
# ════════════════════════════════════════════════════════════════════════════

def plot_similarity_candles_per_seed_temp(
    percents: List[float],
    temperatures: List[float],
    seeds: List[int],
    output_dir: str,
    dedup: bool,
) -> List[str]:
    """One candle chart per (seed, temperature); candle = pairwise Tanimoto spread."""
    out_subdir = os.path.join(output_dir, "candles_similarity_per_seed_temp")
    saved: List[str] = []

    for temperature in temperatures:
        for seed in seeds:
            percent_to_values: Dict[float, List[float]] = {}
            for percent in percents:
                smiles_list = load_valid_generated_smiles(percent, temperature, seed, output_dir, dedup)
                percent_to_values[percent] = (
                    pairwise_tanimoto(smiles_list) if len(smiles_list) >= 2 else []
                )

            dedup_label = "deduplicated" if dedup else "raw (incl. duplicates)"
            path = draw_candle_chart(
                percent_to_values,
                title=(
                    f"Pairwise Tanimoto similarity (T={temperature:g}, seed={seed})\n"
                    f"{dedup_label} generated molecules"
                ),
                ylabel="Pairwise Tanimoto similarity",
                plot_path=os.path.join(
                    out_subdir, f"similarity_candles_T{temperature:g}_seed{seed}.png"
                ),
            )
            if path:
                saved.append(path)
    return saved


# ════════════════════════════════════════════════════════════════════════════
#  PLOT SET 3: pairwise-Tanimoto spread AGGREGATED ACROSS SEEDS  [D2]
# ════════════════════════════════════════════════════════════════════════════

def _pooled_scores(
    percent: float, temperature: float, seeds: List[int], output_dir: str, dedup: bool,
) -> List[float]:
    """'pooled' (D2): merge all seeds' molecules first, then all-pairs once."""
    combined: List[str] = []
    for seed in seeds:
        combined.extend(
            load_valid_generated_smiles(percent, temperature, seed, output_dir, dedup=False)
        )
    if dedup:
        seen: set = set()
        deduped = []
        for smi in combined:
            canon = _canonical_smiles(smi)
            if canon and canon not in seen:
                seen.add(canon)
                deduped.append(canon)
        combined = deduped
    return pairwise_tanimoto(combined) if len(combined) >= 2 else []


def _concat_within_seed_scores(
    percent: float, temperature: float, seeds: List[int], output_dir: str, dedup: bool,
) -> List[float]:
    """'concat' (D2): pairwise within each seed separately, then concatenate scores."""
    all_scores: List[float] = []
    for seed in seeds:
        smiles_list = load_valid_generated_smiles(percent, temperature, seed, output_dir, dedup)
        if len(smiles_list) >= 2:
            all_scores.extend(pairwise_tanimoto(smiles_list))
    return all_scores


_AGGREGATION_LABELS = {
    "pooled": "pooled across seeds (cross-seed pairs included)",
    "concat": "concatenated within-seed pairs only (no cross-seed pairs)",
}


def plot_similarity_candles_aggregated(
    percents: List[float],
    temperatures: List[float],
    seeds: List[int],
    output_dir: str,
    dedup: bool,
    mode: str,
) -> List[str]:
    """
    One candle chart per temperature per aggregation mode.
    mode: "pooled", "concat", or "both" (produces both sets of figures).
    """
    if mode not in ("pooled", "concat", "both"):
        raise ValueError(f"mode must be 'pooled', 'concat', or 'both', got {mode!r}")

    out_subdir = os.path.join(output_dir, "candles_similarity_aggregated")
    saved: List[str] = []
    modes_to_run = ["pooled", "concat"] if mode == "both" else [mode]

    for m in modes_to_run:
        scorer = _pooled_scores if m == "pooled" else _concat_within_seed_scores
        for temperature in temperatures:
            percent_to_values = {
                percent: scorer(percent, temperature, seeds, output_dir, dedup)
                for percent in percents
            }
            path = draw_candle_chart(
                percent_to_values,
                title=(
                    f"Pairwise Tanimoto similarity, aggregated across seeds (T={temperature:g})\n"
                    f"{_AGGREGATION_LABELS[m]}; seeds={seeds}"
                ),
                ylabel="Pairwise Tanimoto similarity",
                plot_path=os.path.join(
                    out_subdir, f"similarity_candles_aggregated_{m}_T{temperature:g}.png"
                ),
            )
            if path:
                saved.append(path)
    return saved


# ════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PROMPTS
# ════════════════════════════════════════════════════════════════════════════

def _ask_aggregation_mode() -> str:
    print("""
  Aggregated similarity plots (one per temperature) can combine the seeds'
  scores two ways:
    pooled : merge all seeds' generated molecules into one pool, compute
             ALL pairs (including cross-seed pairs)
    concat : compute pairwise scores WITHIN each seed separately, then
             concatenate those score lists (never compares molecules
             generated under different seeds)
    both   : produce both sets of figures
""")
    while True:
        raw = input("  Aggregation mode (pooled / concat / both) [both]: ").strip().lower()
        if raw == "":
            return "both"
        if raw in ("pooled", "concat", "both"):
            return raw
        print("    Please type 'pooled', 'concat', 'both', or press Enter for 'both'.")


def _existing_combo_csvs(output_dir: str) -> List[str]:
    return glob.glob(os.path.join(output_dir, "mask*pct_temp*_seed*.csv"))


_STAGE1A_DERIVED_FILES = (
    "rdkit_errors.txt",
    "stage1a_validity_summary.csv",
    "stage1a_validity_vs_mask_percent.png",
)


def _wipe_stage1a_outputs(output_dir: str, existing: List[str]) -> None:
    """Delete existing combo CSVs + derived summary/plot so run_stage1a starts fresh."""
    for path in existing:
        os.remove(path)
    for fname in _STAGE1A_DERIVED_FILES:
        path = os.path.join(output_dir, fname)
        if os.path.isfile(path):
            os.remove(path)
    print(f"  Removed {len(existing)} combo CSV(s) and derived outputs — regenerating from scratch.")


def _ask_reuse_existing_data(output_dir: str) -> bool:
    """
    If Stage 1a data already exists in output_dir, ask whether to reuse it
    (resume/skip already-done molecules — the safe default) or wipe it and
    regenerate everything from scratch. Returns True if run_stage1a() should
    proceed against the EXISTING data (reuse); False after the data has
    already been deleted (fresh run).
    """
    existing = _existing_combo_csvs(output_dir)
    if not existing:
        return True  # nothing to reuse — just run normally

    print(f"\n  Found {len(existing)} existing Stage 1a combo CSV(s) in:\n    {output_dir}")
    if _ask_yes_no_default(
        "Reuse this existing generated data (resume/skip already-done molecules)?",
        default=True,
    ):
        return True

    print(
        f"\n  ⚠️  This will DELETE all {len(existing)} existing combo CSV(s) plus derived "
        f"outputs ({', '.join(_STAGE1A_DERIVED_FILES)}) in:\n    {output_dir}\n"
        f"  before regenerating everything from scratch."
    )
    if _ask_yes_no_default("Are you sure? This cannot be undone.", default=False):
        _wipe_stage1a_outputs(output_dir, existing)
        return True  # data is gone; run_stage1a() now generates fresh

    print("  Keeping existing data instead.")
    return True


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 60)
    print("STAGE 1a EXTENDED: CANDLE-CHART ANALYSIS")
    print("=" * 60)

    # D5/D6: if data from a previous stage1a_random_masking.py run already
    # exists, ask whether to reuse it (default) or wipe it and regenerate.
    # run_stage1a / summarize_and_plot_stage1a are themselves resumable, so
    # either way this reuses whatever's present and fills in the rest.
    _ask_reuse_existing_data(config.STAGE1A_DIR)
    run_stage1a()
    summarize_and_plot_stage1a()

    print("\n" + "─" * 60)
    print("STAGE 1a EXTENDED: CANDLE-CHART OPTIONS")
    print("─" * 60)

    make_seed_candles = _ask_yes_no_default(
        "Make 'unique-valid ratio across seeds' candle charts (one per temperature)?",
        default=True,
    )
    make_similarity_per_seed = _ask_yes_no_default(
        "Make 'pairwise similarity per (seed, temperature)' candle charts?",
        default=True,
    )
    make_similarity_aggregated = _ask_yes_no_default(
        "Make 'pairwise similarity aggregated across seeds' candle charts (one per temperature)?",
        default=True,
    )

    agg_mode = "both"
    if make_similarity_aggregated:
        agg_mode = _ask_aggregation_mode()

    dedup = True
    if make_similarity_per_seed or make_similarity_aggregated:
        print("""
  Deduplicating means: canonicalize each valid generated molecule (RDKit)
  and drop repeats before computing pairwise Tanimoto similarity — the
  same identity rule Stage 1a's own unique-valid-ratio metric uses.
  Default: YES (deduplicate) — matches stage3_analysis.py's convention.
""")
        dedup = _ask_yes_no_default(
            "Deduplicate generated molecules before computing pairwise similarity?",
            default=True,
        )

    percents     = config.STAGE1A_MASK_PERCENTS
    temperatures = config.STAGE1A_TEMPERATURES
    seeds        = config.RANDOM_MASK_SEEDS_LIST
    output_dir   = config.STAGE1A_DIR

    if make_seed_candles:
        plot_unique_valid_seed_candles(percents, temperatures, seeds, output_dir)
    if make_similarity_per_seed:
        plot_similarity_candles_per_seed_temp(percents, temperatures, seeds, output_dir, dedup)
    if make_similarity_aggregated:
        plot_similarity_candles_aggregated(percents, temperatures, seeds, output_dir, dedup, agg_mode)

    print(f"\n  Stage 1a extended complete. Output: {config.STAGE1A_DIR}")


def _run_self_test() -> None:
    """
    Quick smoke test: 5 synthetic molecules, 2 percents, 2 temperatures, 2 seeds.

    The unique-valid-ratio metric always has data (compute_combo_metric
    returns a definite ratio, even 0.0, as long as the combo CSV exists —
    which run_stage1a always writes). Pairwise-similarity candles, however,
    need >=2 valid generated molecules per combo, which is inherently
    stochastic with only a handful of test molecules — so those assertions
    check "produced at least one chart, and never more than the max
    possible," not an exact count.
    """
    import tempfile

    test_molecules = [
        ("TEST_ASPIRIN", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("TEST_CAFFEINE", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("TEST_IBUPROFEN", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ("TEST_PARACETAMOL", "CC(=O)NC1=CC=C(O)C=C1"),
        ("TEST_NAPHTHALENE", "C1=CC2=CC=CC=C2C=C1"),
    ]
    percents     = [10, 20]
    temperatures = [0.5, 1.2]
    seeds        = [17, 53]

    with tempfile.TemporaryDirectory() as td:
        run_stage1a(
            molecules=test_molecules,
            percents=percents,
            temperatures=temperatures,
            seeds=seeds,
            output_dir=td,
        )

        seed_candle_paths = plot_unique_valid_seed_candles(percents, temperatures, seeds, td)
        assert len(seed_candle_paths) == len(temperatures), (
            f"Expected {len(temperatures)} unique-valid candle charts, got {len(seed_candle_paths)}"
        )
        for p in seed_candle_paths:
            assert os.path.isfile(p)

        per_seed_temp_paths = plot_similarity_candles_per_seed_temp(
            percents, temperatures, seeds, td, dedup=True,
        )
        max_per_seed_temp = len(temperatures) * len(seeds)
        assert 1 <= len(per_seed_temp_paths) <= max_per_seed_temp, (
            f"Expected 1-{max_per_seed_temp} per-(seed,T) similarity candles, "
            f"got {len(per_seed_temp_paths)}"
        )
        for p in per_seed_temp_paths:
            assert os.path.isfile(p)
        print(f"  per-(seed,T) similarity candles produced: {len(per_seed_temp_paths)}/{max_per_seed_temp}")

        agg_paths = plot_similarity_candles_aggregated(
            percents, temperatures, seeds, td, dedup=True, mode="both",
        )
        max_agg = 2 * len(temperatures)
        assert 1 <= len(agg_paths) <= max_agg, (
            f"Expected 1-{max_agg} aggregated similarity candles (mode=both), got {len(agg_paths)}"
        )
        for p in agg_paths:
            assert os.path.isfile(p)
        print(f"  aggregated similarity candles produced: {len(agg_paths)}/{max_agg}")

        # Sanity-check the two aggregation definitions actually differ in
        # scope: 'pooled' includes cross-seed pairs (up to 2N molecules -> more
        # pairs) while 'concat' never mixes seeds (two N-molecule pools).
        pooled_scores  = _pooled_scores(percents[0], temperatures[0], seeds, td, dedup=True)
        concat_scores  = _concat_within_seed_scores(percents[0], temperatures[0], seeds, td, dedup=True)
        print(f"  pooled n={len(pooled_scores)}, concat n={len(concat_scores)}")

    print("✅ Stage 1a extended self-test passed.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main()

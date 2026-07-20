# -*- coding: utf-8 -*-
"""
stage1_1a_similarity_score.py
==============================
Stage 1.1a: how similar is each ChemBERTa-generated molecule to the ORIGINAL
molecule it was masked from?

This is a different question from stage1a_random_masking.py's "unique-valid
ratio" (validity + diversity AMONG the generated outputs) and from
stage1a_random_masking_extended.py's pairwise-Tanimoto candles (also
generated-vs-generated). Neither of those ever looks at the original/parent
SMILES. This script does exactly that: for every combo CSV row, it computes
Tanimoto(original_smiles, generated_smiles) — one score per molecule, since
Stage 1a's data model already pairs each row with its own original 1:1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN DECISIONS (resolved with the user; see conversation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
D1. Metric = Tanimoto similarity on Morgan fingerprints (radius=2, 2048
    bits), reusing stage3_analysis._morgan_fp verbatim — same convention as
    every other similarity metric in this project. Each combo CSV row
    already carries both "smiles" (original) and "generated_smiles"
    (candidate) for the SAME molecule, so this is a direct 1:1 comparison,
    not a pool-vs-one-reference comparison like stage3_analysis's
    vs_original_tanimoto.

D2. Invalid generations (valid != "True") are EXCLUDED from the metric —
    Tanimoto is undefined without a fingerprint on both sides. The
    denominator for the summary CSV is therefore "# valid pairs", not
    "# molecules attempted" (unlike stage1a's unique-valid-ratio, which
    divides by total attempted).

D3. Plot style = candle (box-and-whisker) per temperature, x = mask %,
    reusing stage1a_random_masking_extended.draw_candle_chart verbatim.
    Two candle sets are produced:
      (a) candles_similarity_vs_original/ — one figure per temperature;
          each box = per-molecule similarity scores POOLED across all
          seeds (matches how stage1a's own line-plot metric pools/means
          across seeds, but shown as a full distribution instead of one
          mean point).
      (b) candles_similarity_vs_original_by_seed/ — one figure per
          temperature; each box = per-seed MEAN similarity (only
          len(seeds) points per box), mirroring
          stage1a_random_masking_extended.plot_unique_valid_seed_candles —
          shows how sensitive the vs-original similarity is to which
          tokens happened to get masked (seed).

D4. Data source: reuses existing combo CSVs in config.STAGE1A_DIR as-is
    (no dedup — every row is a distinct original molecule, so there is no
    cross-row identity to collapse). If NO combo CSVs exist at all, an
    interactive prompt (default: yes) offers to run
    stage1a_random_masking.run_stage1a() + summarize_and_plot_stage1a()
    first. A combo CSV missing for just some (percent, temperature, seed)
    points is skipped with a warning, matching
    stage1a_random_masking.aggregate_stage1a_results' convention.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage1_1a_similarity_score.py

HOW TO TEST (quick smoke test, no full ChEMBL download needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage1_1a_similarity_score.py --test
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List

from rdkit.DataStructs import TanimotoSimilarity

import config
from stage1a_random_masking import combo_csv_path, run_stage1a, summarize_and_plot_stage1a
from stage1a_random_masking_extended import draw_candle_chart
from stage3_analysis import _ask_yes_no_default, _morgan_fp


# ════════════════════════════════════════════════════════════════════════════
#  METRIC: per-molecule Tanimoto(original, generated)  [D1, D2]
# ════════════════════════════════════════════════════════════════════════════

def load_vs_original_scores(
    percent: float,
    temperature: float,
    seed: int,
    output_dir: str,
) -> List[float]:
    """
    Read one combo CSV and return one Tanimoto(original, generated) score
    per row where the generated molecule was RDKit-valid.

    Rows with valid != "True" are skipped (D2) — there is no fingerprint to
    compare. A row is also skipped (silently) if the ORIGINAL smiles itself
    fails to fingerprint, which should not happen since Stage 0a already
    RDKit-verified every original SMILES.
    """
    csv_path = combo_csv_path(output_dir, percent, temperature, seed)
    if not os.path.isfile(csv_path):
        return []

    scores: List[float] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("valid") != "True" or not row.get("generated_smiles"):
                continue
            orig_fp = _morgan_fp(row["smiles"])
            gen_fp = _morgan_fp(row["generated_smiles"])
            if orig_fp is not None and gen_fp is not None:
                scores.append(TanimotoSimilarity(orig_fp, gen_fp))
    return scores


# ════════════════════════════════════════════════════════════════════════════
#  PLOT SET A: vs-original similarity, POOLED across seeds  [D3a]
# ════════════════════════════════════════════════════════════════════════════

def plot_vs_original_candles_pooled(
    percents: List[float],
    temperatures: List[float],
    seeds: List[int],
    output_dir: str,
) -> List[str]:
    """One candle chart per temperature; box = per-molecule scores pooled across seeds."""
    out_subdir = os.path.join(output_dir, "candles_similarity_vs_original")
    saved: List[str] = []

    for temperature in temperatures:
        percent_to_values: Dict[float, List[float]] = {}
        for percent in percents:
            pooled: List[float] = []
            for seed in seeds:
                pooled.extend(load_vs_original_scores(percent, temperature, seed, output_dir))
            percent_to_values[percent] = pooled

        path = draw_candle_chart(
            percent_to_values,
            title=(
                f"Similarity to original molecule (T={temperature:g})\n"
                f"pooled across seeds={seeds}; invalid generations excluded"
            ),
            ylabel="Tanimoto(original, generated)\n(one point per valid molecule)",
            plot_path=os.path.join(out_subdir, f"vs_original_similarity_candles_T{temperature:g}.png"),
        )
        if path:
            saved.append(path)
    return saved


# ════════════════════════════════════════════════════════════════════════════
#  PLOT SET B: vs-original similarity spread ACROSS SEEDS  [D3b]
# ════════════════════════════════════════════════════════════════════════════

def plot_vs_original_candles_by_seed(
    percents: List[float],
    temperatures: List[float],
    seeds: List[int],
    output_dir: str,
) -> List[str]:
    """One candle chart per temperature; box = per-seed MEAN score (len(seeds) points per box)."""
    out_subdir = os.path.join(output_dir, "candles_similarity_vs_original_by_seed")
    saved: List[str] = []

    for temperature in temperatures:
        percent_to_values: Dict[float, List[float]] = {}
        for percent in percents:
            seed_means: List[float] = []
            for seed in seeds:
                scores = load_vs_original_scores(percent, temperature, seed, output_dir)
                if scores:
                    seed_means.append(sum(scores) / len(scores))
            percent_to_values[percent] = seed_means

        path = draw_candle_chart(
            percent_to_values,
            title=(
                f"Similarity to original molecule: spread across seeds (T={temperature:g})\n"
                f"seeds={seeds}"
            ),
            ylabel="Mean Tanimoto(original, generated)\n(one point per seed)",
            plot_path=os.path.join(out_subdir, f"vs_original_similarity_seed_spread_T{temperature:g}.png"),
        )
        if path:
            saved.append(path)
    return saved


# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY CSV
# ════════════════════════════════════════════════════════════════════════════

def write_vs_original_summary_csv(
    percents: List[float],
    temperatures: List[float],
    seeds: List[int],
    output_dir: str,
    output_path: str,
) -> str:
    """Per (temperature, percent): mean similarity + n_valid_pairs, pooled across seeds."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["temperature", "mask_percent", "mean_similarity_vs_original", "n_valid_pairs"])
        for temperature in temperatures:
            for percent in percents:
                pooled: List[float] = []
                for seed in seeds:
                    pooled.extend(load_vs_original_scores(percent, temperature, seed, output_dir))
                mean_sim = sum(pooled) / len(pooled) if pooled else ""
                writer.writerow([temperature, percent, mean_sim, len(pooled)])
    return output_path


# ════════════════════════════════════════════════════════════════════════════
#  DATA AVAILABILITY  [D4]
# ════════════════════════════════════════════════════════════════════════════

def _ensure_stage1a_data(output_dir: str) -> None:
    """If no Stage 1a combo CSVs exist yet, offer to generate them first."""
    existing = glob.glob(os.path.join(output_dir, "mask*pct_temp*_seed*.csv"))
    if existing:
        return

    print(f"\n  No Stage 1a combo CSVs found in:\n    {output_dir}")
    if _ask_yes_no_default(
        "Run stage1a_random_masking.py generation now to produce them?",
        default=True,
    ):
        run_stage1a()
        summarize_and_plot_stage1a()
    else:
        print("  Proceeding without data — plots/summary will be empty where combos are missing.")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 60)
    print("STAGE 1.1a: SIMILARITY TO ORIGINAL MOLECULE")
    print("=" * 60)

    output_dir   = config.STAGE1A_DIR
    percents     = config.STAGE1A_MASK_PERCENTS
    temperatures = config.STAGE1A_TEMPERATURES
    seeds        = config.RANDOM_MASK_SEEDS_LIST

    _ensure_stage1a_data(output_dir)

    write_vs_original_summary_csv(
        percents, temperatures, seeds, output_dir,
        os.path.join(output_dir, "stage1_1a_vs_original_similarity_summary.csv"),
    )
    plot_vs_original_candles_pooled(percents, temperatures, seeds, output_dir)
    plot_vs_original_candles_by_seed(percents, temperatures, seeds, output_dir)

    print(f"\n  Stage 1.1a complete. Output: {output_dir}")


def _run_self_test() -> None:
    """Quick smoke test: 5 synthetic molecules, 2 percents, 2 temperatures, 2 seeds."""
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

        # Every valid-row score must be a real Tanimoto value in [0, 1].
        any_scores = False
        for percent in percents:
            for temperature in temperatures:
                for seed in seeds:
                    scores = load_vs_original_scores(percent, temperature, seed, td)
                    for s in scores:
                        assert 0.0 <= s <= 1.0, f"Similarity out of range: {s}"
                    any_scores = any_scores or bool(scores)
        assert any_scores, "Expected at least one valid vs-original score across the whole sweep"

        summary_path = write_vs_original_summary_csv(
            percents, temperatures, seeds, td,
            os.path.join(td, "stage1_1a_vs_original_similarity_summary.csv"),
        )
        assert os.path.isfile(summary_path)
        with open(summary_path, newline="", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))
        assert len(summary_rows) == len(percents) * len(temperatures), (
            f"Expected {len(percents) * len(temperatures)} summary rows, got {len(summary_rows)}"
        )

        pooled_paths = plot_vs_original_candles_pooled(percents, temperatures, seeds, td)
        assert 1 <= len(pooled_paths) <= len(temperatures)
        for p in pooled_paths:
            assert os.path.isfile(p)

        by_seed_paths = plot_vs_original_candles_by_seed(percents, temperatures, seeds, td)
        assert 1 <= len(by_seed_paths) <= len(temperatures)
        for p in by_seed_paths:
            assert os.path.isfile(p)

    print("✅ Stage 1.1a self-test passed.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main()

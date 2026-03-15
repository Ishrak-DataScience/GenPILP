# -*- coding: utf-8 -*-
"""
stage4_br4_matching.py
======================
Stage 4 of the pipeline — BR4 nearest-neighbour analysis.

For each group of generated molecules (one group = one source ligand, e.g.
EAM-A-1), finds the single most chemically similar molecule from the BR4
reference set (BR4_PDB_Data.csv) and answers:
  "Are the generated molecules mostly close to one particular BR4 ligand?"

Per group, produces:
  similarity_score_histogram.png
      x = Tanimoto similarity to nearest BR4 neighbour  (0–1)
      y = count of generated molecules
      Shows how close the generated molecules are to anything in BR4.

  nearest_neighbour_frequency.png
      x = BR4 ligand code (deduplicated by SMILES, Ambiguity 5 Option A)
      y = count of generated molecules whose nearest BR4 neighbour is that ligand
      Shows which BR4 ligand dominates the nearest-neighbour mapping.

  closeness_summary.txt
      Human-readable decision statement, e.g.:
        "All 12/12 generated molecules are most similar to BR4 ligand JQ1
         (Tanimoto ≥ 0.40, median similarity = 0.73)"
        or
        "7/12 (58%) generated molecules are most similar to BR4 ligand JQ1
         (Tanimoto ≥ 0.40).  5/12 have no BR4 match above threshold."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1.  Similarity metric: Tanimoto on Morgan fingerprints (radius=2, 2048 bits)
     via rdFingerprintGenerator — no deprecation warnings.

A2.  BR4 CSV deduplication: rows with identical SMILES strings are treated as
     the same ligand. The display label is the first `Ligand Code` encountered
     for that SMILES (Ambiguity 5 Option A).

A3.  BR4 rows whose SMILES has fewer than min_heavy_atoms heavy atoms are
     excluded before any comparison. min_heavy_atoms is asked at runtime
     (default 7). Heavy atom count = mol.GetNumHeavyAtoms() via RDKit.

A4.  For each generated molecule, the nearest BR4 neighbour is the single
     BR4 ligand with the highest Tanimoto similarity. Ties broken by first
     occurrence in the (deduplicated) BR4 list.

A5.  The closeness percentage uses denominator = all generated molecules in
     the group (Interpretation P). Molecules whose nearest-neighbour similarity
     is < T are reported as "no BR4 match above threshold."

A6.  Generated molecules are read from config.PRED_DIR sub-folders, same
     structure as Stage 3. Pool choice (IA / random / both) is asked at
     runtime (Ambiguity 1 Option B).

A7.  If a generated SMILES cannot be parsed by RDKit it is silently skipped
     (consistent with Stage 3 Assumption A7).

A8.  If a BR4 SMILES cannot be parsed by RDKit, that row is silently skipped
     with a warning printed once at load time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python stage4_br4_matching.py

Must be run AFTER stage2_molecule_generation.py.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Answer "yes" to the smoke-test prompt when running the file, or call:
      python stage4_br4_matching.py
  and type "yes" at the first prompt.

  The test writes to config.STAGE4_DIR/test/ (wiped each run).
"""

import csv
import glob
import io
import json
import os
import shutil
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

import config
from stage3_analysis import (
    _read_smiles_from_txt,
    _safe,
    load_ligand_predictions,
    pool_smiles,
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


def _ask_pool_choice() -> str:
    """Ask which pool of generated molecules to use. Returns 'ia', 'rand', or 'both'."""
    print("""
  Pool choice — which generated molecules to analyse:
    ia   : interaction-aware only  (Stage-1 masks)
    rand : random only             (Stage-1.5 masks)
    both : IA + random, deduplicated  [default]
""")
    while True:
        raw = input("  Pool choice (ia / rand / both) [both]: ").strip().lower()
        if raw == "":
            return "both"
        if raw in ("ia", "rand", "both"):
            return raw
        print("    Please type 'ia', 'rand', or 'both'.")


def _ask_min_heavy_atoms() -> int:
    """Ask minimum heavy atom count for BR4 filter. Default 7."""
    while True:
        raw = input("  Minimum heavy atoms for BR4 ligands (filters solvents) [7]: ").strip()
        if raw == "":
            return 7
        try:
            val = int(raw)
            if val >= 0:
                return val
        except ValueError:
            pass
        print("    Please type a non-negative integer.")


def _ask_threshold() -> float:
    """Ask Tanimoto similarity threshold T. Default 0.4."""
    while True:
        raw = input("  Similarity threshold T for 'close' decision (0.0–1.0) [0.4]: ").strip()
        if raw == "":
            return 0.4
        try:
            val = float(raw)
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
        print("    Please type a number between 0.0 and 1.0.")


def ask_runtime_options() -> Tuple[str, int, float]:
    """
    Ask all runtime options in sequence.
    Returns (pool_choice, min_heavy_atoms, threshold).
    """
    print("\n" + "─" * 60)
    print("STAGE 4 OPTIONS")
    print("─" * 60)

    pool        = _ask_pool_choice()
    min_heavy   = _ask_min_heavy_atoms()
    threshold   = _ask_threshold()

    print(f"\n  Settings confirmed:")
    print(f"    Pool              : {pool}")
    print(f"    Min heavy atoms   : {min_heavy}")
    print(f"    Threshold T       : {threshold}")
    print("─" * 60 + "\n")
    return pool, min_heavy, threshold


# ════════════════════════════════════════════════════════════════════════════
#  FINGERPRINT HELPER
# ════════════════════════════════════════════════════════════════════════════

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fp(smiles: str):
    """Return Morgan fingerprint or None if SMILES is unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _FP_GEN.GetFingerprint(mol)


# ════════════════════════════════════════════════════════════════════════════
#  BR4 CSV LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_br4_ligands(
    csv_path: str,
    min_heavy_atoms: int = 7,
) -> List[Tuple[str, str]]:
    """
    Load BR4 reference ligands from csv_path.

    Deduplicates by SMILES (Assumption A2).
    Filters rows whose SMILES has fewer than min_heavy_atoms heavy atoms
    (Assumption A3).
    Skips rows with unparseable SMILES with a warning (Assumption A8).

    Returns
    -------
    List of (label, smiles) tuples where label = first Ligand Code seen
    for that SMILES.
    """
    df = pd.read_csv(csv_path)

    # Locate the SMILES and Ligand Code columns robustly
    smiles_col = next(
        (c for c in df.columns if c.strip().lower() in ("smiles", "smile")), None
    )
    label_col  = next(
        (c for c in df.columns if c.strip().lower() in ("ligand code", "ligand_code")), None
    )
    if smiles_col is None:
        raise ValueError(f"Cannot find a 'Smiles' column in {csv_path}. "
                         f"Columns found: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Cannot find a 'Ligand Code' column in {csv_path}. "
                         f"Columns found: {list(df.columns)}")

    seen_smiles: dict = {}     # canonical_smiles → label
    skipped_invalid  = 0
    skipped_small    = 0

    for _, row in df.iterrows():
        raw_smi = str(row[smiles_col]).strip()
        label   = str(row[label_col]).strip()

        mol = Chem.MolFromSmiles(raw_smi)
        if mol is None:
            skipped_invalid += 1
            continue

        if mol.GetNumHeavyAtoms() < min_heavy_atoms:
            skipped_small += 1
            continue

        canon = Chem.MolToSmiles(mol)   # canonical form as dedup key
        if canon not in seen_smiles:
            seen_smiles[canon] = label

    if skipped_invalid:
        print(f"  ⚠️  BR4 load: {skipped_invalid} row(s) skipped (unparseable SMILES).")
    if skipped_small:
        print(f"  ℹ️  BR4 load: {skipped_small} row(s) filtered out "
              f"(< {min_heavy_atoms} heavy atoms).")

    result = [(label, canon) for canon, label in seen_smiles.items()]
    print(f"  ✅ BR4 reference set: {len(result)} unique ligands loaded "
          f"(min_heavy_atoms={min_heavy_atoms}).")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  NEAREST-NEIGHBOUR SEARCH
# ════════════════════════════════════════════════════════════════════════════

def find_nearest_br4(
    generated_smiles: List[str],
    br4_ligands: List[Tuple[str, str]],
) -> List[Tuple[str, str, float]]:
    """
    For each generated SMILES find its single nearest BR4 neighbour.

    Parameters
    ----------
    generated_smiles : list of generated SMILES strings
    br4_ligands      : list of (label, smiles) from load_br4_ligands()

    Returns
    -------
    List of (generated_smiles, nearest_br4_label, tanimoto_score).
    Generated molecules whose fingerprint cannot be computed are omitted
    (Assumption A7).
    """
    # Pre-compute BR4 fingerprints once
    br4_fps = []
    for label, smi in br4_ligands:
        fp = _fp(smi)
        if fp is not None:
            br4_fps.append((label, fp))

    results = []
    for gen_smi in generated_smiles:
        gen_fp = _fp(gen_smi)
        if gen_fp is None:
            continue

        best_label = None
        best_score = -1.0
        for label, br4_fp in br4_fps:
            score = TanimotoSimilarity(gen_fp, br4_fp)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is not None:
            results.append((gen_smi, best_label, float(best_score)))

    return results


# ════════════════════════════════════════════════════════════════════════════
#  CLOSENESS DECISION STATEMENT
# ════════════════════════════════════════════════════════════════════════════

def build_closeness_summary(
    nn_results: List[Tuple[str, str, float]],
    threshold: float,
    lig_id: str,
) -> str:
    """
    Build the human-readable closeness decision statement.

    Parameters
    ----------
    nn_results  : output of find_nearest_br4()
    threshold   : T — minimum similarity to count as "close"
    lig_id      : source ligand identifier (for display)

    Returns
    -------
    Multi-line string summary.
    """
    n_total = len(nn_results)
    if n_total == 0:
        return f"  No generated molecules could be compared for {lig_id}.\n"

    # Molecules meeting threshold (Interpretation P denominator = all generated)
    above = [(smi, label, score)
             for smi, label, score in nn_results if score >= threshold]
    n_above  = len(above)
    n_below  = n_total - n_above
    fraction = n_above / n_total

    lines = [f"  Source ligand  : {lig_id}"]
    lines.append(f"  Generated mols : {n_total}")
    lines.append(f"  Threshold T    : {threshold:.2f}")
    lines.append("")

    if n_above == 0:
        lines.append(
            f"  ✗ None of the {n_total} generated molecules have a BR4 "
            f"nearest neighbour with Tanimoto ≥ {threshold:.2f}."
        )
        return "\n".join(lines) + "\n"

    # Dominant BR4 ligand among those above threshold
    label_counts = Counter(label for _, label, _ in above)
    dominant_label, dominant_count = label_counts.most_common(1)[0]
    dominant_scores = [score for _, label, score in above
                       if label == dominant_label]
    median_sim = float(np.median(dominant_scores))

    # All nearest neighbours the same?
    all_same = (len(label_counts) == 1)

    if all_same and n_above == n_total:
        lines.append(
            f"  ✓ All {n_total}/{n_total} generated molecules are most similar "
            f"to BR4 ligand '{dominant_label}' "
            f"(Tanimoto ≥ {threshold:.2f}, median similarity = {median_sim:.3f})."
        )
    elif all_same:
        lines.append(
            f"  ✓ {n_above}/{n_total} ({fraction*100:.0f}%) generated molecules "
            f"are most similar to BR4 ligand '{dominant_label}' "
            f"(Tanimoto ≥ {threshold:.2f}, median similarity = {median_sim:.3f})."
        )
        lines.append(
            f"    {n_below}/{n_total} have no BR4 match above threshold."
        )
    else:
        lines.append(
            f"  ✓ {n_above}/{n_total} ({fraction*100:.0f}%) generated molecules "
            f"have a BR4 nearest neighbour with Tanimoto ≥ {threshold:.2f}."
        )
        lines.append(
            f"    Dominant BR4 ligand: '{dominant_label}' "
            f"({dominant_count}/{n_above} of those above threshold, "
            f"median similarity = {median_sim:.3f})."
        )
        if n_below:
            lines.append(
                f"    {n_below}/{n_total} have no BR4 match above threshold."
            )
        # List all BR4 ligands that appear
        lines.append("    Full breakdown (above threshold only):")
        for label, count in label_counts.most_common():
            scores = [s for _, l, s in above if l == label]
            lines.append(
                f"      '{label}': {count} mol(s), "
                f"median similarity = {np.median(scores):.3f}"
            )

    return "\n".join(lines) + "\n"


# ════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_similarity_histogram(
    nn_results: List[Tuple[str, str, float]],
    threshold: float,
    lig_id: str,
    out_path: str,
) -> bool:
    """
    Histogram: x = Tanimoto to nearest BR4 neighbour, y = count of generated mols.
    Vertical dashed line at threshold T.
    """
    if not nn_results:
        print(f"    ⚠️  No data for similarity histogram of '{lig_id}'. Skipped.")
        return False

    scores = [score for _, _, score in nn_results]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=min(20, max(5, len(scores))),
            color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.8,
               label=f"threshold T = {threshold:.2f}")
    ax.set_xlabel("Tanimoto Similarity to Nearest BR4 Neighbour", fontsize=12)
    ax.set_ylabel("Count of Generated Molecules", fontsize=12)
    ax.set_title(
        f"BR4 Nearest-Neighbour Similarity — {lig_id}\n"
        f"(n = {len(scores)} molecules)",
        fontsize=12,
    )
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_nn_frequency(
    nn_results: List[Tuple[str, str, float]],
    threshold: float,
    lig_id: str,
    out_path: str,
) -> bool:
    """
    Bar chart: x = BR4 ligand code, y = count of generated mols whose
    nearest neighbour is that ligand (all molecules, not just above threshold).
    Bars above-threshold molecules are coloured differently.
    """
    if not nn_results:
        print(f"    ⚠️  No data for frequency chart of '{lig_id}'. Skipped.")
        return False

    # Count all nearest neighbours regardless of threshold
    all_counts    = Counter(label for _, label, _ in nn_results)
    # Count only those above threshold per label
    above_counts  = Counter(label for _, label, score in nn_results
                            if score >= threshold)

    labels_sorted = sorted(all_counts.keys(),
                           key=lambda l: all_counts[l], reverse=True)
    x       = range(len(labels_sorted))
    all_y   = [all_counts[l] for l in labels_sorted]
    above_y = [above_counts.get(l, 0) for l in labels_sorted]
    below_y = [all_y[i] - above_y[i] for i in range(len(labels_sorted))]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(labels_sorted) * 1.2), 5))

    ax.bar(x, above_y, color="#4CAF50", label=f"Tanimoto ≥ {threshold:.2f}",
           alpha=0.9)
    ax.bar(x, below_y, bottom=above_y, color="#FF9800",
           label=f"Tanimoto < {threshold:.2f}", alpha=0.9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("BR4 Ligand Code (nearest neighbour)", fontsize=12)
    ax.set_ylabel("Count of Generated Molecules", fontsize=12)
    ax.set_title(
        f"Nearest BR4 Neighbour Frequency — {lig_id}\n"
        f"(n = {len(nn_results)} molecules)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ════════════════════════════════════════════════════════════════════════════
#  POOL LOADING  (uses Stage 3 helpers)
# ════════════════════════════════════════════════════════════════════════════

def load_pool_for_ligand(
    lig_dir: str,
    pool_choice: str,
) -> List[str]:
    """
    Load generated SMILES for one ligand folder according to pool_choice.

    pool_choice : 'ia'   → ia_mask*.txt only
                  'rand' → rand_mask*.txt only
                  'both' → IA + random, deduplicated
    """
    predictions = load_ligand_predictions(lig_dir)
    mask_counts = sorted(predictions.keys())

    if pool_choice == "ia":
        all_smi = [s for mc in mask_counts for s in predictions[mc]["ia"]]
    elif pool_choice == "rand":
        all_smi = [s for mc in mask_counts for s in predictions[mc]["rand"]]
    else:   # "both"
        ia   = [s for mc in mask_counts for s in predictions[mc]["ia"]]
        rand = [s for mc in mask_counts for s in predictions[mc]["rand"]]
        all_smi = pool_smiles(ia, rand, deduplicate=True)

    # Validate each SMILES (Assumption A7)
    valid = [s for s in all_smi if Chem.MolFromSmiles(s) is not None]
    return valid


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_stage4(
    pred_dir:        Optional[str] = None,
    csv_path:        Optional[str] = None,
    stage4_dir:      Optional[str] = None,
    pool_choice:     str = "both",
    min_heavy_atoms: int = 7,
    threshold:       float = 0.4,
) -> Dict[str, dict]:
    """
    Run Stage-4 BR4 nearest-neighbour analysis.

    Returns
    -------
    Dict keyed by ligand_id:
        {
          "n_generated"  : int,
          "nn_results"   : [(gen_smi, br4_label, score), ...],
          "summary_text" : str,
          "sim_hist_ok"  : bool,
          "freq_chart_ok": bool,
        }
    """
    pred_dir   = pred_dir   or config.PRED_DIR
    csv_path   = csv_path   or config.REF_CSV_PATH
    stage4_dir = stage4_dir or config.STAGE4_DIR

    os.makedirs(stage4_dir, exist_ok=True)

    # ── Load BR4 reference set ────────────────────────────────────────────────
    print(f"\n  Loading BR4 reference set from:\n    {csv_path}")
    br4_ligands = load_br4_ligands(csv_path, min_heavy_atoms=min_heavy_atoms)
    if not br4_ligands:
        print("  ❌ BR4 reference set is empty after filtering. Exiting.")
        return {}

    # ── Discover ligand folders ───────────────────────────────────────────────
    lig_dirs = sorted([
        d for d in glob.glob(os.path.join(pred_dir, "*"))
        if os.path.isdir(d)
    ])
    if not lig_dirs:
        print(f"  ❌ No ligand sub-folders found in {pred_dir}")
        print("     Run stage2_molecule_generation.py first.")
        return {}
    print(f"  Ligand folders found: {len(lig_dirs)}")

    all_results: Dict[str, dict] = {}

    for lig_dir in lig_dirs:
        lig_id  = os.path.basename(lig_dir)
        out_dir = os.path.join(stage4_dir, _safe(lig_id))
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Ligand: {lig_id}  |  pool: {pool_choice}")
        print(f"{'='*60}")

        # ── Load generated pool ───────────────────────────────────────────────
        pool = load_pool_for_ligand(lig_dir, pool_choice)
        print(f"  Generated molecules in pool: {len(pool)}")

        if not pool:
            print("  ⚠️  Empty pool. Skipping.")
            continue

        # ── Nearest-neighbour search ──────────────────────────────────────────
        nn_results = find_nearest_br4(pool, br4_ligands)
        print(f"  Nearest-neighbour pairs computed: {len(nn_results)}")

        # ── Closeness summary ─────────────────────────────────────────────────
        summary = build_closeness_summary(nn_results, threshold, lig_id)
        print(summary)

        summary_path = os.path.join(out_dir, "closeness_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Stage 4 — BR4 Nearest-Neighbour Closeness Summary\n")
            f.write(f"Pool choice     : {pool_choice}\n")
            f.write(f"Min heavy atoms : {min_heavy_atoms}\n\n")
            f.write(summary)
        print(f"  💾 Summary saved: {summary_path}")

        # ── Plots ─────────────────────────────────────────────────────────────
        sim_ok = plot_similarity_histogram(
            nn_results  = nn_results,
            threshold   = threshold,
            lig_id      = lig_id,
            out_path    = os.path.join(out_dir, "similarity_score_histogram.png"),
        )
        freq_ok = plot_nn_frequency(
            nn_results  = nn_results,
            threshold   = threshold,
            lig_id      = lig_id,
            out_path    = os.path.join(out_dir, "nearest_neighbour_frequency.png"),
        )
        if sim_ok:
            print(f"  🖼  similarity_score_histogram.png saved.")
        if freq_ok:
            print(f"  🖼  nearest_neighbour_frequency.png saved.")

        all_results[lig_id] = {
            "n_generated":   len(pool),
            "nn_results":    nn_results,
            "summary_text":  summary,
            "sim_hist_ok":   sim_ok,
            "freq_chart_ok": freq_ok,
        }

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED TEST
# ════════════════════════════════════════════════════════════════════════════

_TEST_GENERATED = [
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "OC(=O)c1ccccc1O",                # salicylic acid
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",    # caffeine
    "c1ccc2ccccc2c1",                  # naphthalene
]

_TEST_BR4_CSV_ROWS = [
    ("PDB ID", "Ligand Code", "Ligand Chain", "UniProt ID",
     "Smiles", "Lig_ChEMBL_ID", "Chembl ID"),
    # water and ethylene glycol — should be filtered at min_heavy=7
    ("FAKE", "HOH", "A", "O60885", "O",    "X", "Y"),
    ("FAKE", "EDO", "A", "O60885", "OCCO", "X", "Y"),
    # real drug-like molecules
    ("FAKE", "JQ1",  "A", "O60885",
     "CC1=C(C)C(=C(C(=O)NC2=CC=C(C=C2)S(N)(=O)=O)C(F)(F)F)C=C1", "X", "Y"),
    ("FAKE", "ASP",  "A", "O60885",
     "CC(=O)Oc1ccccc1C(=O)O",   "X", "Y"),   # aspirin — should be top match
    ("FAKE", "CAF",  "A", "O60885",
     "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "X", "Y"),  # caffeine — matches one gen mol
]


def _run_test() -> bool:
    import tempfile

    print("\n" + "=" * 60)
    print("STAGE 4 SELF-TEST")
    print("=" * 60)

    td = os.path.join(config.STAGE4_DIR, "test")
    if os.path.exists(td):
        shutil.rmtree(td)
        print(f"  🗑  Wiped existing test dir: {td}")
    os.makedirs(td)
    print(f"  📁 Created fresh test dir:  {td}\n")

    pred_dir   = os.path.join(td, "preds")
    stage4_dir = os.path.join(td, "stage4_out")

    # ── Write synthetic Stage-2 prediction txt files ──────────────────────────
    lig_id  = "EAM-A-1"
    lig_dir = os.path.join(pred_dir, _safe(lig_id))
    os.makedirs(lig_dir)

    for mc in range(1, 4):
        with open(os.path.join(lig_dir, f"ia_mask{mc:03d}.txt"), "w") as f:
            f.write("\n".join(_TEST_GENERATED) + "\n")
        with open(os.path.join(lig_dir, f"rand_mask{mc:03d}.txt"), "w") as f:
            f.write("\n".join(_TEST_GENERATED[:3]) + "\n")

    # ── Write synthetic BR4 CSV ───────────────────────────────────────────────
    csv_path = os.path.join(td, "test_br4.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in _TEST_BR4_CSV_ROWS:
            writer.writerow(row)

    passed = 0
    failed = 0

    def check(condition: bool, msg: str):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ PASS  {msg}")
            passed += 1
        else:
            print(f"  ❌ FAIL  {msg}")
            failed += 1

    # ── Run Stage 4 ───────────────────────────────────────────────────────────
    results = run_stage4(
        pred_dir        = pred_dir,
        csv_path        = csv_path,
        stage4_dir      = stage4_dir,
        pool_choice     = "both",
        min_heavy_atoms = 7,
        threshold       = 0.4,
    )

    check(lig_id in results,
          f"'{lig_id}' present in results")

    if lig_id in results:
        r = results[lig_id]

        check(r["n_generated"] > 0,
              f"pool is non-empty (got {r['n_generated']})")
        check(len(r["nn_results"]) > 0,
              f"nn_results non-empty")
        check(r["sim_hist_ok"],
              "similarity_score_histogram.png produced")
        check(r["freq_chart_ok"],
              "nearest_neighbour_frequency.png produced")

        # All nn_results should reference ASP or CAF (not HOH/EDO — filtered)
        br4_labels_used = {label for _, label, _ in r["nn_results"]}
        check("HOH" not in br4_labels_used and "EDO" not in br4_labels_used,
              "solvents HOH/EDO correctly filtered out of BR4 set")

        # Check output files exist on disk
        out_dir = os.path.join(stage4_dir, _safe(lig_id))
        for fname in ["similarity_score_histogram.png",
                      "nearest_neighbour_frequency.png",
                      "closeness_summary.txt"]:
            check(os.path.exists(os.path.join(out_dir, fname)),
                  f"{fname} on disk")

        # Scores should be in [0, 1]
        scores = [score for _, _, score in r["nn_results"]]
        check(all(0.0 <= s <= 1.0 for s in scores),
              "all similarity scores in [0, 1]")

        # ASP (aspirin) should be top nearest neighbour for aspirin molecule
        asp_nn = [(label, score) for gen, label, score in r["nn_results"]
                  if gen == "CC(=O)Oc1ccccc1C(=O)O"]
        if asp_nn:
            check(asp_nn[0][0] == "ASP",
                  f"aspirin generated mol → nearest neighbour is 'ASP' "
                  f"(got '{asp_nn[0][0]}')")

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
    print("STAGE 4: BR4 NEAREST-NEIGHBOUR MATCHING")
    print("=" * 60)

    print(f"""
  This stage compares each generated molecule to the BR4 reference set
  ({config.REF_CSV_PATH})
  and finds the most chemically similar BR4 ligand for each generated molecule.

  Before running the full analysis you can run a smoke test instead.

  The smoke test:
    • Does NOT need real Stage-2 prediction files or the BR4 CSV.
    • Creates synthetic predictions for one ligand (EAM-A-1) and a
      small synthetic BR4 CSV (aspirin, caffeine, JQ1 + 2 solvents).
    • Runs the full nearest-neighbour search, plots, and summary.
    • Verifies solvents are filtered, output files exist, and scores
      are in range.
    • Saves everything to:
        {os.path.join(config.STAGE4_DIR, "test")}
      (wiped clean at the start of every test run).
""")

    run_test = _ask_yes_no_default("Run the smoke test?", default=False)
    if run_test:
        ok = _run_test()
        sys.exit(0 if ok else 1)

    pool, min_heavy, threshold = ask_runtime_options()

    results = run_stage4(
        pool_choice     = pool,
        min_heavy_atoms = min_heavy,
        threshold       = threshold,
    )

    print("\n" + "=" * 60)
    print("✅ Stage 4 complete.")
    print(f"   Outputs saved to: {config.STAGE4_DIR}")
    print(f"   Ligands analysed: {len(results)}")
    for lig_id, r in sorted(results.items()):
        print(
            f"   • {lig_id}  n={r['n_generated']}  "
            f"sim_hist={'✓' if r['sim_hist_ok'] else '✗'}  "
            f"freq_chart={'✓' if r['freq_chart_ok'] else '✗'}"
        )


if __name__ == "__main__":
    main()

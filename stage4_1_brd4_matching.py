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

  nearest_neighbour_frequency.png (UPDATED)
      x = BR4 ligand code (deduplicated by SMILES, Ambiguity 5 Option A)
      y = count of generated molecules whose nearest BR4 neighbour is that ligand
      Shows which BR4 ligand dominates the nearest-neighbour mapping.
      Bars are colored by similarity score bins, with the median annotated on top.

  nearest_neighbour_distribution.png (NEW)
      Boxplot showing the actual distribution of similarity scores.

  closeness_summary.txt
      Human-readable decision statement.
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

def _ask_generator_stage() -> str:
    """
    Ask which generator stage produced the molecules to analyse.
    Returns one of: "2", "2.5", "2.7"

    Maps to:
      2   → config.PRED_DIR          (Stage 2 prefix-slice)
      2.5 → config.STAGE25_PRED_DIR  (Stage 2.5 random-pick, single seed)
      2.7 → config.STAGE27_PRED_DIR  (Stage 2.7 multi-seed aggregated)
    """
    print(f"""
  Generator stage — which predicted molecules do you want to analyse?

    2   : Stage 2   — prefix-slice masking
          {config.PRED_DIR}
    2.5 : Stage 2.5 — random-pick, single seed
          {config.STAGE25_PRED_DIR}
    2.7 : Stage 2.7 — multi-seed aggregated
          {config.STAGE27_PRED_DIR}
""")
    valid = {"2", "2.5", "2.7"}
    while True:
        raw = input("  Generator stage (2 / 2.5 / 2.7) [2]: ").strip()
        if raw == "":
            return "2"
        if raw in valid:
            return raw
        print("    Please type 2, 2.5, or 2.7.")


def _resolve_pred_dir(gen_stage: str) -> str:
    """Return the predictions directory for the given generator stage."""
    return {
        "2":   config.PRED_DIR,
        "2.5": config.STAGE25_PRED_DIR,
        "2.7": config.STAGE27_PRED_DIR,
    }[gen_stage]


def ask_runtime_options() -> Tuple[str, int, float]:
    print("\n" + "─" * 60)
    print("STAGE 4 OPTIONS")
    print("─" * 60)
    gen_stage   = _ask_generator_stage()
    pool        = _ask_pool_choice()
    min_heavy   = _ask_min_heavy_atoms()
    threshold   = _ask_threshold()
    print(f"\n  Settings confirmed:")
    print(f"    Generator stage   : {gen_stage}")
    print(f"    Pool              : {pool}")
    print(f"    Min heavy atoms   : {min_heavy}")
    print(f"    Threshold T       : {threshold}")
    print("─" * 60 + "\n")
    return gen_stage, pool, min_heavy, threshold

# ════════════════════════════════════════════════════════════════════════════
#  FINGERPRINT HELPER
# ════════════════════════════════════════════════════════════════════════════

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def _fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _FP_GEN.GetFingerprint(mol)

# ════════════════════════════════════════════════════════════════════════════
#  BR4 CSV LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_br4_ligands(csv_path: str, min_heavy_atoms: int = 7) -> List[Tuple[str, str]]:
    df = pd.read_csv(csv_path)
    smiles_col = next((c for c in df.columns if c.strip().lower() in ("smiles", "smile")), None)
    label_col  = next((c for c in df.columns if c.strip().lower() in ("ligand code", "ligand_code")), None)
    
    seen_smiles: dict = {}
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
        canon = Chem.MolToSmiles(mol)
        if canon not in seen_smiles:
            seen_smiles[canon] = label

    result = [(label, canon) for canon, label in seen_smiles.items()]
    return result

# ════════════════════════════════════════════════════════════════════════════
#  NEAREST-NEIGHBOUR SEARCH
# ════════════════════════════════════════════════════════════════════════════

def find_nearest_br4(generated_smiles: List[str], br4_ligands: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
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

def load_pool_for_ligand(lig_dir: str, pool_choice: str) -> List[str]:
    pool = []
    if pool_choice in ("ia", "both"):
        ia_files = glob.glob(os.path.join(lig_dir, "ia_mask*.txt"))
        for f in ia_files:
            pool.extend(_read_smiles_from_txt(f))
    if pool_choice in ("rand", "both"):
        rand_files = glob.glob(os.path.join(lig_dir, "rand_mask*.txt"))
        for f in rand_files:
            pool.extend(_read_smiles_from_txt(f))
    return list(set(pool))

# ════════════════════════════════════════════════════════════════════════════
#  CLOSENESS DECISION STATEMENT
# ════════════════════════════════════════════════════════════════════════════

def build_closeness_summary(nn_results: List[Tuple[str, str, float]], threshold: float, lig_id: str) -> str:
    n_total = len(nn_results)
    if n_total == 0:
        return f"  No generated molecules could be compared for {lig_id}.\n"

    above = [(smi, label, score) for smi, label, score in nn_results if score >= threshold]
    n_above = len(above)
    n_below = n_total - n_above
    fraction = n_above / n_total if n_total > 0 else 0

    lines = [f"  Source ligand  : {lig_id}"]
    lines.append(f"  Generated mols : {n_total}")
    lines.append(f"  Threshold T    : {threshold:.2f}")
    lines.append("")

    if n_above == 0:
        lines.append(f"  ✗ None of the {n_total} generated molecules have a match with Tanimoto >= {threshold:.2f}.")
        return "\n".join(lines) + "\n"

    label_counts = Counter(label for _, label, _ in above)
    dominant_label, dominant_count = label_counts.most_common(1)[0]
    dominant_scores = [score for _, label, score in above if label == dominant_label]
    median_sim = float(np.median(dominant_scores))
    all_same = (len(label_counts) == 1)

    if all_same and n_above == n_total:
        lines.append(f"  ✓ All {n_total}/{n_total} generated molecules are most similar to '{dominant_label}' (T >= {threshold:.2f}, med = {median_sim:.3f}).")
    elif all_same:
        lines.append(f"  ✓ {n_above}/{n_total} ({fraction*100:.0f}%) generated molecules are most similar to '{dominant_label}' (T >= {threshold:.2f}, med = {median_sim:.3f}).")
        lines.append(f"    {n_below}/{n_total} have no match above threshold.")
    else:
        lines.append(f"  ✓ {n_above}/{n_total} ({fraction*100:.0f}%) generated molecules have a nearest neighbour with T >= {threshold:.2f}.")
        lines.append(f"    Dominant ligand: '{dominant_label}' ({dominant_count}/{n_above} of those above threshold, median similarity = {median_sim:.3f}).")
        if n_below:
            lines.append(f"    {n_below}/{n_total} have no match above threshold.")
        lines.append("    Full breakdown (above threshold only):")
        for lbl, count in label_counts.most_common():
            lbl_scores = [s for _, l, s in above if l == lbl]
            lbl_med = float(np.median(lbl_scores))
            lines.append(f"      '{lbl}': {count} mol(s), median similarity = {lbl_med:.3f}")
            
    return "\n".join(lines) + "\n"

# ════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_similarity_histogram(nn_results: List[Tuple[str, str, float]], threshold: float, lig_id: str, out_path: str) -> bool:
    if not nn_results: return False
    scores = [score for _, _, score in nn_results]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(scores, bins=np.linspace(0, 1, 21), color="#4FC3F7", edgecolor="black")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")
    ax.set_title(f"Nearest-Neighbour Similarity — {lig_id}")
    ax.set_xlabel("Tanimoto Similarity")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True

def plot_nn_frequency(
    nn_results: List[Tuple[str, str, float]],
    threshold: float,
    lig_id: str,
    out_path: str,
) -> bool:
    """
    Stacked bar chart of nearest-neighbour frequencies (above threshold).
    Bars are colored by similarity score bins, with the median annotated on top.
    """
    above = [(label, score) for smi, label, score in nn_results if score >= threshold]
    if not above:
        return False

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    from collections import defaultdict
    label_scores = defaultdict(list)
    for label, score in above:
        label_scores[label].append(score)

    labels = sorted(label_scores.keys(), key=lambda k: len(label_scores[k]), reverse=True)

    bins = [
        (threshold, 0.6, "#FFD54F", f"Low ({threshold:.2f} - 0.60)"),
        (0.6, 0.8, "#81C784", "Medium (0.60 - 0.80)"),
        (0.8, 1.01, "#388E3C", "High (0.80 - 1.00)")
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(labels))

    for idx, label in enumerate(labels):
        scores = label_scores[label]
        median_val = float(np.median(scores))
        total_height = 0

        # Draw stacked bars
        for b_min, b_max, color, _ in bins:
            count = sum(1 for s in scores if b_min <= s < b_max)
            if count > 0:
                ax.bar(x_positions[idx], count, bottom=total_height, color=color, edgecolor="black", width=0.6)
                total_height += count

        # --> OPTION 1: Add Median Text directly hovering over the bar
        ax.text(
            x_positions[idx], 
            total_height + (max(1, total_height * 0.02)), 
            f"Med: {median_val:.2f}", 
            ha='center', 
            va='bottom', 
            fontsize=9, 
            fontweight='bold', 
            rotation=45,  
            color='black'
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Count of Generated Molecules")
    ax.set_title(f"Nearest-Neighbour Frequency (T ≥ {threshold:.2f}) — {lig_id}")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor='black', label=l) for _, _, c, l in bins]
    ax.legend(handles=legend_elements, title="Similarity Range")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True

def plot_nn_distribution_boxplot(
    nn_results: List[Tuple[str, str, float]],
    threshold: float,
    lig_id: str,
    out_path: str,
) -> bool:
    """
    Boxplot showing the actual distribution of similarity scores (Option 2).
    """
    above = [(label, score) for smi, label, score in nn_results if score >= threshold]
    if not above:
        return False

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    from collections import defaultdict
    label_scores = defaultdict(list)
    for label, score in above:
        label_scores[label].append(score)

    labels = sorted(label_scores.keys(), key=lambda k: len(label_scores[k]), reverse=True)
    data_to_plot = [label_scores[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=labels)

    for patch in box['boxes']:
        patch.set_facecolor('#bbdefb')
    for median in box['medians']:
        median.set(color='red', linewidth=2)

    ax.set_ylabel("Tanimoto Similarity Score")
    ax.set_title(f"Similarity Distribution by Nearest-Neighbour (T ≥ {threshold:.2f}) — {lig_id}")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Threshold")
    
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True

def run_stage4(
    pool_choice:    str   = "both",
    min_heavy_atoms: int  = 7,
    threshold:      float = 0.4,
    gen_stage:      str   = "2",
) -> Dict[str, dict]:
    """
    Run Stage-4 BR4 nearest-neighbour analysis.

    Parameters
    ----------
    gen_stage : generator stage whose predictions to analyse.
                "2"   → config.PRED_DIR
                "2.5" → config.STAGE25_PRED_DIR
                "2.7" → config.STAGE27_PRED_DIR

    Output directory (Option A — nested by generator stage):
                config.STAGE4_DIR / stage{gen_stage} / <lig_id> /
    """
    pred_dir   = _resolve_pred_dir(gen_stage)
    # Nested output: preserve results from different generator stages side by side
    stage4_dir = os.path.join(config.STAGE4_DIR, f"stage{gen_stage}", pool_choice)
    os.makedirs(stage4_dir, exist_ok=True)

    print(f"  Generator stage : Stage {gen_stage}")
    print(f"  pred_dir        : {pred_dir}")
    print(f"  Output dir      : {stage4_dir}")

    br4_ligands = load_br4_ligands(config.REF_CSV_PATH, min_heavy_atoms=min_heavy_atoms)

    lig_dirs = sorted([d for d in glob.glob(os.path.join(pred_dir, "*")) if os.path.isdir(d)])
    all_results: Dict[str, dict] = {}
    for lig_dir in lig_dirs:
        lig_id  = os.path.basename(lig_dir)
        out_dir = os.path.join(stage4_dir, _safe(lig_id))
        os.makedirs(out_dir, exist_ok=True)
        
        pool = load_pool_for_ligand(lig_dir, pool_choice)
        nn_results = find_nearest_br4(pool, br4_ligands)
        summary = build_closeness_summary(nn_results, threshold, lig_id)
        
        with open(os.path.join(out_dir, "closeness_summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary)
            
        plot_similarity_histogram(nn_results, threshold, lig_id, os.path.join(out_dir, "similarity_score_histogram.png"))
        plot_nn_frequency(nn_results, threshold, lig_id, os.path.join(out_dir, "nearest_neighbour_frequency.png"))
        plot_nn_distribution_boxplot(nn_results, threshold, lig_id, os.path.join(out_dir, "nearest_neighbour_distribution.png"))
        
        all_results[lig_id] = {"n_generated": len(pool), "nn_results": nn_results, "summary_text": summary}
    return all_results

def main():
    gen_stage, pool, min_heavy, threshold = ask_runtime_options()
    run_stage4(
        pool_choice     = pool,
        min_heavy_atoms = min_heavy,
        threshold       = threshold,
        gen_stage       = gen_stage,
    )

if __name__ == "__main__":
    main()
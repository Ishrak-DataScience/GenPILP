# -*- coding: utf-8 -*-
"""
stage8_novelty_potency_analysis.py
====================================
Stage 8 of the pipeline — Novelty vs Potency scatter plot, Pareto front
analysis, hierarchical filtering, and top-10 diverse potent molecule report.

Reads docking_summary.csv from Stage 6 and Stage-1 .meta.json files (for
original ligand SMILES).  For each generated molecule computes:
  • Tanimoto similarity to the original ligand (X axis, 0–1)
  • A user-chosen docking score (Y axis)

Then:
  1. Scatter plot — per ligand group AND combined (all groups, colour-coded)
  2. Hierarchical filter — CNNscore > T1, Vinardo < T2, sorted by CNNaffinity
  3. Composite score   — CNNaffinity × CNNscore
  4. Pareto front      — non-dominated set balancing composite score and novelty
  5. Top-10 PDF report — most novel + highest score molecules from Pareto front,
                         with molecule drawing, scores, and Tanimoto per row

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1.  Input CSV: config.STAGE8_INPUT_CSV (full path).
     If absent or not found on disk, the user is prompted to enter a path.
     Prompt also triggered if user provides a path at runtime.

A2.  Tanimoto computed using Morgan fingerprints r=2, 2048 bits against the
     original ligand SMILES from Stage-1 .meta.json (same method as Stage 3/4).

A3.  Pareto front: two objectives — composite score (higher=better) and
     novelty = 1 − Tanimoto (higher=better, i.e. lower Tanimoto).
     A molecule is on the front if no other molecule is simultaneously better
     on BOTH objectives (strictly dominates on at least one).

A4.  Top-10 from Pareto front: sorted by composite score descending.
     If the front has fewer than 10 molecules, all are shown.

A5.  Hierarchical filter thresholds are asked at runtime with scientific
     defaults: CNNscore > 0.70, Vinardo < −7.0 kcal/mol.

A6.  The Y-axis docking score for the scatter is asked at runtime:
     CNNscore / CNNaffinity / Vinardo / composite (CNNaffinity×CNNscore).

A7.  Only pose 1 per molecule is used (GNINA top-ranked pose).

A8.  Top-10 PDF uses reportlab (same dependency as Stage 7).
     Each row contains: Rank | Molecule drawing | SMILES | CNNaffinity |
     CNNscore | Vinardo | Tanimoto | Composite score

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install reportlab      # if not already installed
  python stage8_novelty_potency_analysis.py

Must be run after stage6_docking.py.

HOW TO TEST (no real docking data required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Answer "yes" to the smoke-test prompt.
"""

import glob
import io
import json
import os
import shutil
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity

import config

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

SCORE_COLS = ["CNNaffinity", "CNNscore", "minimizedAffinity"]

# Group colours for combined scatter plot
_GROUP_COLORS = [
    "#1F77B4", "#FF7F0E", "#2CA02C",
    "#D62728", "#9467BD", "#8C564B",
]

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


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


def _ask_csv_path() -> str:
    """Prompt user for a docking_summary.csv path."""
    print(f"\n  config.STAGE8_INPUT_CSV = {config.STAGE8_INPUT_CSV}")
    while True:
        raw = input(
            "  Path to docking_summary.csv  "
            "(press Enter to use config value): "
        ).strip()
        path = raw if raw else config.STAGE8_INPUT_CSV
        if os.path.isfile(path):
            return path
        print(f"    ❌ File not found: {path}")
        print("       Please enter the full path to a docking_summary.csv file.")


def _resolve_csv_path() -> str:
    """
    Use config.STAGE8_INPUT_CSV if it exists on disk.
    Otherwise fall through to the prompt.
    """
    cfg_path = config.STAGE8_INPUT_CSV
    if os.path.isfile(cfg_path):
        print(f"  ✅ Using config CSV: {cfg_path}")
        return cfg_path
    print(f"  ⚠️  config.STAGE8_INPUT_CSV not found on disk: {cfg_path}")
    return _ask_csv_path()


def _ask_y_score() -> str:
    """Ask which score to use on the Y axis."""
    print("""
  Y-axis docking score for scatter plot:
    1. CNNaffinity              — CNN-predicted -log Kd/Ki  (higher = better)  [default]
    2. CNNscore                 — CNN binding probability   (higher = better)
    3. minimizedAffinity        — Vinardo score kcal/mol    (lower  = better)
    4. composite                — CNNaffinity × CNNscore    (higher = better)
""")
    mapping = {
        "1": "CNNaffinity", "2": "CNNscore",
        "3": "minimizedAffinity", "4": "composite",
        "cnnaffinity": "CNNaffinity", "cnnscore": "CNNscore",
        "minimizedaffinity": "minimizedAffinity",
        "vinardo": "minimizedAffinity", "composite": "composite",
    }
    while True:
        raw = input("  Y-axis choice (1/2/3/4) [1]: ").strip().lower()
        if raw == "":
            return "CNNaffinity"
        if raw in mapping:
            return mapping[raw]
        print("    Please enter 1, 2, 3, or 4.")


def _ask_hierarchical_thresholds() -> Tuple[float, float]:
    """
    Ask CNNscore and Vinardo thresholds for hierarchical filtering.
    Returns (cnn_score_min, vinardo_max).
    """
    print("""
  Hierarchical filter thresholds:
    Step 1 — keep molecules with CNNscore  > threshold  (removes misplaced poses)
    Step 2 — keep molecules with Vinardo   < threshold  (ensures physical energy)
    Step 3 — sort survivors by CNNaffinity

  Scientific defaults:
    Standard stringency : CNNscore > 0.70,  Vinardo < -7.0
    High    stringency  : CNNscore > 0.80,  Vinardo < -8.0
""")
    # CNNscore threshold
    while True:
        raw = input("  CNNscore threshold (0.0–1.0) [0.70]: ").strip()
        if raw == "":
            cnn_t = 0.70
            break
        try:
            v = float(raw)
            if 0.0 <= v <= 1.0:
                cnn_t = v
                break
        except ValueError:
            pass
        print("    Please enter a number between 0.0 and 1.0.")

    # Vinardo threshold
    while True:
        raw = input("  Vinardo threshold (kcal/mol) [-7.0]: ").strip()
        if raw == "":
            vin_t = -7.0
            break
        try:
            vin_t = float(raw)
            break
        except ValueError:
            print("    Please enter a number (e.g. -7.0 or -8.0).")

    print(f"\n  Filter: CNNscore > {cnn_t}  AND  Vinardo < {vin_t}")
    return cnn_t, vin_t


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_original_smiles(mask_calc_dir: str) -> Dict[str, str]:
    """Load {ligand_key → canonical SMILES} from Stage-1 .meta.json files."""
    result: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(mask_calc_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            lig = meta.get("ligand", {})
            key = (f"{lig.get('resname','?')}-"
                   f"{lig.get('chain','?')}-"
                   f"{lig.get('resseq','?')}")
            smi = meta.get("smiles", "")
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                result[key] = Chem.MolToSmiles(mol)
        except Exception as e:
            print(f"  ⚠️  Could not read {os.path.basename(path)}: {e}")
    return result


def load_docking_data(
    csv_path: str,
    original_smiles_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Load pose-1 docking results, coerce scores, compute composite score
    and Tanimoto similarity to the original ligand per group.

    Returns a DataFrame with added columns:
      composite   — CNNaffinity × CNNscore
      tanimoto    — Tanimoto to original ligand (NaN if original unavailable)
      novelty     — 1 − tanimoto
    """
    # Try UTF-8 first; fall back to latin-1 which decodes every byte 0x00-0xFF
    # without error and is a superset of Windows-1252 (cp1252).
    # This handles CSVs saved by Excel or Windows tools with smart-quote encoding.
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"  ⚠️  CSV is not UTF-8 — retrying with latin-1 encoding.")
        df = pd.read_csv(csv_path, encoding="latin-1")
    for c in SCORE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Pose 1 only (Assumption A7)
    df = df[df["pose"] == 1].copy().reset_index(drop=True)

    # Composite score
    df["composite"] = df["CNNaffinity"] * df["CNNscore"]

    # Tanimoto per row
    tanimoto_vals = []
    for _, row in df.iterrows():
        lig_id  = row["ligand_group"]
        gen_smi = str(row["smiles"])
        orig    = original_smiles_map.get(lig_id)
        if orig is None:
            tanimoto_vals.append(float("nan"))
            continue
        mol_gen  = Chem.MolFromSmiles(gen_smi)
        mol_orig = Chem.MolFromSmiles(orig)
        if mol_gen is None or mol_orig is None:
            tanimoto_vals.append(float("nan"))
            continue
        fp_gen  = _FP_GEN.GetFingerprint(mol_gen)
        fp_orig = _FP_GEN.GetFingerprint(mol_orig)
        tanimoto_vals.append(float(TanimotoSimilarity(fp_gen, fp_orig)))

    df["tanimoto"] = tanimoto_vals
    df["novelty"]  = 1.0 - df["tanimoto"]
    return df


def _ask_show_pareto() -> bool:
    """Ask whether to overlay Pareto-front points on the scatter plots."""
    return _ask_yes_no_default(
        "Show Pareto-front molecules highlighted on scatter plots?",
        default=False,
    )


def _ask_ranking_methods() -> set:
    """
    Ask which top-10 ranking methods to run. User can select multiple.

    Returns a set of method keys from:
      {"idea1", "idea3", "idea4", "idea5"}
    """
    print("""
  ─────────────────────────────────────────────────────────────
  TOP-10 RANKING METHODS
  Select one or more (comma-separated, e.g. "1,4,5").
  ─────────────────────────────────────────────────────────────

  1 — Option A / Idea 1: Tanimoto cutoff + composite rank  [DEFAULT]
      Keep molecules with Tanimoto < cutoff (novel enough),
      rank survivors by CNNaffinity × CNNscore, take top 10.
      Trade-off: simple & interpretable, but cutoff is arbitrary.

  3 — Idea 3: Hierarchical filter → Pareto front
      Apply CNNscore + Vinardo quality gates first, then find
      the Pareto front on survivors (composite vs novelty),
      take top 10 from the front by composite.
      Trade-off: scientifically principled, fewest arbitrary choices;
      may yield < 10 if few molecules survive the filter.

  4 — Idea 4: MMR (Maximal Marginal Relevance)
      Iteratively pick molecules maximising:
        λ × composite − (1−λ) × max_tanimoto_to_already_selected
      Guarantees the top-10 are mutually diverse (not just novel
      vs original). λ=0.5 balances potency and diversity.
      Trade-off: less common in VS papers; O(N²) but fast for N<1000.

  5 — Idea 5: Butina cluster-pick
      Cluster all molecules by Tanimoto (Butina algorithm),
      pick the highest-composite molecule per cluster,
      sort representatives by composite, take top 10.
      Trade-off: guarantees one representative per scaffold;
      number of clusters depends on cutoff.
""")
    valid = {"1", "3", "4", "5"}
    key_map = {"1": "idea1", "3": "idea3", "4": "idea4", "5": "idea5"}
    while True:
        raw = input("  Methods (1/3/4/5, comma-separated) [1]: ").strip()
        if raw == "":
            return {"idea1"}
        parts = {p.strip() for p in raw.split(",")}
        if parts.issubset(valid):
            return {key_map[p] for p in parts}
        print(f"    Please enter numbers from 1,3,4,5 only.")


def _ask_tanimoto_cutoff() -> float:
    """Ask Tanimoto cutoff for Idea 1 (Option A)."""
    print("""
  Tanimoto cutoff for Idea 1 (Option A):
    Molecules with Tanimoto < cutoff are considered novel enough.
    Typical values: 0.30 (strict novelty), 0.40 (standard), 0.60 (loose).
""")
    while True:
        raw = input("  Tanimoto cutoff (0.0–1.0) [0.40]: ").strip()
        if raw == "":
            return 0.40
        try:
            v = float(raw)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
        print("    Please enter a number between 0.0 and 1.0.")


def _ask_mmr_lambda() -> float:
    """Ask MMR λ for Idea 4. Default 0.5."""
    print("""
  MMR λ (Idea 4):
    λ = 1.0 → pure potency  (same result as Idea 1 without cutoff)
    λ = 0.5 → equal weight potency + diversity  [default]
    λ = 0.0 → pure diversity
""")
    while True:
        raw = input("  λ value (0.0–1.0) [0.5]: ").strip()
        if raw == "":
            return 0.5
        try:
            v = float(raw)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
        print("    Please enter a number between 0.0 and 1.0.")


def _ask_cluster_cutoff() -> float:
    """Ask Butina clustering cutoff for Idea 5. Default 0.4."""
    print("""
  Butina clustering cutoff (Idea 5):
    Lower = more clusters (finer structural grouping).
    Higher = fewer clusters (coarser grouping).
    Standard value: 0.4 (Bemis-Murcko convention).
""")
    while True:
        raw = input("  Clustering cutoff (0.0–1.0) [0.4]: ").strip()
        if raw == "":
            return 0.4
        try:
            v = float(raw)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
        print("    Please enter a number between 0.0 and 1.0.")


# ════════════════════════════════════════════════════════════════════════════
#  PARETO FRONT
# ════════════════════════════════════════════════════════════════════════════

def pareto_front(
    df: pd.DataFrame,
    score_col: str = "composite",
) -> pd.DataFrame:
    """
    Compute the Pareto front for two objectives:
      1. score_col  — higher is better  (potency)
      2. novelty    — higher is better  (= 1 − tanimoto)

    A point is on the front if no other point is simultaneously ≥ on BOTH
    objectives AND strictly > on at least one.

    Returns the subset of df that is on the Pareto front.
    """
    df_valid = df.dropna(subset=[score_col, "novelty"]).copy()
    scores   = df_valid[score_col].values
    novelty  = df_valid["novelty"].values
    n        = len(df_valid)
    on_front = np.ones(n, dtype=bool)

    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j or not on_front[j]:
                continue
            # j dominates i if j is >= on both and strictly > on at least one
            if (scores[j] >= scores[i] and novelty[j] >= novelty[i] and
                    (scores[j] > scores[i] or novelty[j] > novelty[i])):
                on_front[i] = False
                break

    return df_valid.iloc[on_front].copy()


# ════════════════════════════════════════════════════════════════════════════
#  HIERARCHICAL FILTER
# ════════════════════════════════════════════════════════════════════════════

def hierarchical_filter(
    df: pd.DataFrame,
    cnn_score_min: float = 0.70,
    vinardo_max:   float = -7.0,
) -> pd.DataFrame:
    """
    Step 1: CNNscore > cnn_score_min   (removes incorrectly posed compounds)
    Step 2: Vinardo  < vinardo_max     (ensures physical binding energy)
    Step 3: sort by CNNaffinity descending

    Returns filtered + sorted DataFrame.
    """
    step1 = df[df["CNNscore"] > cnn_score_min].copy()
    step2 = step1[step1["minimizedAffinity"] < vinardo_max].copy()
    step3 = step2.sort_values("CNNaffinity", ascending=False).reset_index(drop=True)

    print(f"  Hierarchical filter:")
    print(f"    Input          : {len(df)} molecules")
    print(f"    After CNNscore > {cnn_score_min} : {len(step1)}")
    print(f"    After Vinardo  < {vinardo_max}   : {len(step2)}")
    return step3


# ════════════════════════════════════════════════════════════════════════════
#  TOP-10 SELECTION METHODS
# ════════════════════════════════════════════════════════════════════════════

def top10_idea1(
    df: pd.DataFrame,
    tanimoto_cutoff: float = 0.40,
    cnn_score_min:   float = 0.70,
    vinardo_max:     float = -7.0,
) -> pd.DataFrame:
    """
    Idea 1 — all 4 steps applied in order:
      Step 1: CNNscore > cnn_score_min  — remove incorrectly posed compounds
      Step 2: Vinardo  < vinardo_max    — remove weak physical binders
      Step 3: Tanimoto < tanimoto_cutoff — remove scaffold-similar molecules
      Step 4: Sort by composite (CNNaffinity × CNNscore) descending, top 10
    """
    step1 = df[df["CNNscore"] > cnn_score_min].copy()
    step2 = step1[step1["minimizedAffinity"] < vinardo_max].copy()
    step3 = step2[step2["tanimoto"] < tanimoto_cutoff].copy()
    result = step3.sort_values("composite", ascending=False).head(10).reset_index(drop=True)
    print(f"  Idea 1:")
    print(f"    Input                          : {len(df)}")
    print(f"    After CNNscore > {cnn_score_min}        : {len(step1)}")
    print(f"    After Vinardo  < {vinardo_max}     : {len(step2)}")
    print(f"    After Tanimoto < {tanimoto_cutoff}        : {len(step3)}")
    print(f"    Top 10 by composite            : {len(result)}")
    return result


def top10_idea3(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    cnn_score_min: float = 0.70,
    vinardo_max:   float = -7.0,
) -> pd.DataFrame:
    """
    Idea 3: Apply hierarchical quality filter first (CNNscore + Vinardo),
    then compute the Pareto front on survivors, return top 10 by composite.
    """
    step1 = df[df["CNNscore"] > cnn_score_min]
    step2 = step1[step1["minimizedAffinity"] < vinardo_max].copy()
    if step2.empty:
        print(f"  Idea 3: no molecules survived filter (CNNscore>{cnn_score_min}, Vinardo<{vinardo_max})")
        return pd.DataFrame()
    pf = pareto_front(step2, score_col="composite")
    result = pf.sort_values("composite", ascending=False).head(10).reset_index(drop=True)
    print(f"  Idea 3: {len(df)} → {len(step2)} after filter → {len(pf)} on Pareto front → top {len(result)}")
    return result


def top10_idea4_mmr(df: pd.DataFrame, lam: float = 0.5) -> pd.DataFrame:
    """
    Idea 4: Maximal Marginal Relevance (MMR).

    Iteratively selects molecules maximising:
      score = λ × composite_normalised
              − (1−λ) × max_tanimoto_to_already_selected

    λ=1.0 = pure potency; λ=0.5 = balanced; λ=0.0 = pure diversity.
    Returns up to 10 molecules.
    """
    valid = df.dropna(subset=["composite", "tanimoto"]).copy()
    if valid.empty:
        return pd.DataFrame()

    # Normalise composite to [0, 1]
    c_min, c_max = valid["composite"].min(), valid["composite"].max()
    if c_max == c_min:
        valid["_comp_norm"] = 1.0
    else:
        valid["_comp_norm"] = (valid["composite"] - c_min) / (c_max - c_min)

    # Pre-compute fingerprints
    fps = []
    for smi in valid["smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        fps.append(_FP_GEN.GetFingerprint(mol) if mol else None)
    valid = valid.reset_index(drop=True)

    selected_idx: List[int] = []
    remaining    = list(range(len(valid)))

    for _ in range(min(10, len(valid))):
        best_score = -np.inf
        best_i     = None
        for i in remaining:
            comp_norm = valid.at[i, "_comp_norm"]
            if not selected_idx:
                max_sim = 0.0
            else:
                sims = []
                for j in selected_idx:
                    if fps[i] is not None and fps[j] is not None:
                        sims.append(TanimotoSimilarity(fps[i], fps[j]))
                max_sim = max(sims) if sims else 0.0
            score = lam * comp_norm - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected_idx.append(best_i)
        remaining.remove(best_i)

    result = valid.iloc[selected_idx].copy().reset_index(drop=True)
    print(f"  Idea 4 MMR (λ={lam}): selected {len(result)} molecules")
    return result


def top10_idea5_cluster(df: pd.DataFrame, cluster_cutoff: float = 0.4) -> pd.DataFrame:
    """
    Idea 5: Butina clustering → pick best-composite molecule per cluster →
    sort representatives by composite → top 10.
    """
    from rdkit.ML.Cluster import Butina
    from rdkit.Chem import DataStructs

    valid = df.dropna(subset=["composite", "tanimoto"]).copy().reset_index(drop=True)
    if valid.empty:
        return pd.DataFrame()

    fps = []
    for smi in valid["smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        fps.append(_FP_GEN.GetFingerprint(mol) if mol else None)

    # Build distance matrix (upper triangle) — Butina expects 1 − Tanimoto
    dists = []
    n = len(fps)
    for i in range(1, n):
        for j in range(i):
            if fps[i] is not None and fps[j] is not None:
                dists.append(1.0 - TanimotoSimilarity(fps[i], fps[j]))
            else:
                dists.append(1.0)

    clusters = Butina.ClusterData(dists, n, cluster_cutoff, isDistData=True)

    # Pick best composite from each cluster
    picks = []
    for cluster in clusters:
        cluster_df = valid.iloc[list(cluster)]
        best = cluster_df.loc[cluster_df["composite"].idxmax()]
        picks.append(best)

    result = (pd.DataFrame(picks)
              .sort_values("composite", ascending=False)
              .head(10)
              .reset_index(drop=True))
    print(f"  Idea 5 Cluster (cutoff={cluster_cutoff}): "
          f"{n} molecules → {len(clusters)} clusters → top {len(result)}")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  SCATTER PLOTS
# ════════════════════════════════════════════════════════════════════════════

def _scatter_one_group(
    df_group:   pd.DataFrame,
    lig_id:     str,
    y_col:      str,
    pareto_df:  pd.DataFrame,
    out_path:   str,
    show_pareto: bool = False,
) -> None:
    """One scatter plot for a single ligand group."""
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#FAFAFA")

    y_lower_better = (y_col == "minimizedAffinity")

    # All molecules
    ax.scatter(
        df_group["tanimoto"], df_group[y_col],
        c="#5B8DB8", alpha=0.55, s=35, linewidths=0,
        label="All docked molecules",
    )

    # Pareto front subset — shown only if requested
    pf_group = pareto_df[pareto_df["ligand_group"] == lig_id]
    if show_pareto and len(pf_group) > 0:
        ax.scatter(
            pf_group["tanimoto"], pf_group[y_col],
            c="#D4A017", edgecolors="#8B6A00", linewidths=0.8,
            s=80, zorder=5, label=f"Pareto front (n={len(pf_group)})",
        )
        # Annotate top-3 Pareto points by composite
        top3 = pf_group.nlargest(3, "composite")
        for rank_i, (_, r) in enumerate(top3.iterrows(), start=1):
            ax.annotate(
                f"#{rank_i}",
                xy=(r["tanimoto"], r[y_col]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8, color="#8B0000", fontweight="bold",
            )

    # Reference lines
    ax.axvline(0.4, color="#AAAAAA", linestyle="--", linewidth=0.8,
               label="Tanimoto = 0.40 (scaffold similarity boundary)")

    y_label = {
        "CNNaffinity":       "CNN Affinity  (higher = better)",
        "CNNscore":          "CNN Score  (higher = better)",
        "minimizedAffinity": "Vinardo  kcal/mol  (lower = better)",
        "composite":         "CNNaffinity × CNNscore  (higher = better)",
    }.get(y_col, y_col)

    ax.set_xlabel("Tanimoto Similarity to Original Ligand", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(
        f"Novelty vs Potency — {lig_id}\n"
        f"← more novel | more similar →   (n = {len(df_group)})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    if y_lower_better:
        ax.invert_yaxis()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼  {os.path.basename(out_path)}")


def _scatter_combined(
    df: pd.DataFrame,
    y_col: str,
    pareto_df: pd.DataFrame,
    out_path: str,
    show_pareto: bool = False,
) -> None:
    """One combined scatter plot — all groups colour-coded."""
    groups = sorted(df["ligand_group"].unique())
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#FAFAFA")

    y_lower_better = (y_col == "minimizedAffinity")

    for gi, grp in enumerate(groups):
        col      = _GROUP_COLORS[gi % len(_GROUP_COLORS)]
        df_grp   = df[df["ligand_group"] == grp]
        pf_grp   = pareto_df[pareto_df["ligand_group"] == grp]

        ax.scatter(
            df_grp["tanimoto"], df_grp[y_col],
            c=col, alpha=0.45, s=30, linewidths=0, label=grp,
        )
        if show_pareto and len(pf_grp) > 0:
            ax.scatter(
                pf_grp["tanimoto"], pf_grp[y_col],
                c=col, edgecolors="black", linewidths=0.8,
                s=90, zorder=5, marker="*",
            )

    ax.axvline(0.4, color="#AAAAAA", linestyle="--", linewidth=0.8,
               label="Tanimoto = 0.40")

    y_label = {
        "CNNaffinity":       "CNN Affinity  (higher = better)",
        "CNNscore":          "CNN Score  (higher = better)",
        "minimizedAffinity": "Vinardo  kcal/mol  (lower = better)",
        "composite":         "CNNaffinity × CNNscore  (higher = better)",
    }.get(y_col, y_col)

    handles, labels = ax.get_legend_handles_labels()
    if show_pareto:
        star_patch = mpatches.Patch(
            facecolor="white", edgecolor="black",
            label="★ Pareto front molecules",
        )
        handles = handles + [star_patch]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.85, ncol=2)

    ax.set_xlabel("Tanimoto Similarity to Original Ligand", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(
        f"Novelty vs Potency — All Groups Combined\n"
        f"(n = {len(df)} molecules, {len(groups)} ligand groups)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    if y_lower_better:
        ax.invert_yaxis()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼  {os.path.basename(out_path)}")


# ════════════════════════════════════════════════════════════════════════════
#  TOP-10 PDF REPORT
# ════════════════════════════════════════════════════════════════════════════

def _mol_png_bytes(smi: str, w: int = 200, h: int = 160) -> Optional[bytes]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(w, h))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_top10_pdf(
    pareto_df:    pd.DataFrame,
    out_path:     str,
    title_note:   str = "",
    method_label: str = "Pareto Front",
) -> None:
    """
    Build a PDF report of top-10 molecules, one row per molecule with
    drawing + all scores. method_label identifies which selection method
    was used (shown in the PDF title).

    Columns:
      Rank | Drawing | SMILES | CNNaffinity | CNNscore | Vinardo |
      Tanimoto | Composite
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image as RLImage, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )

    top10 = (pareto_df
             .sort_values("composite", ascending=False)
             .head(10)
             .reset_index(drop=True))

    if len(top10) == 0:
        print("  ⚠️  No Pareto-front molecules to write to PDF.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    page_w, page_h = landscape(A3)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=landscape(A3),
        leftMargin=1.0 * cm, rightMargin=1.0 * cm,
        topMargin=1.0 * cm, bottomMargin=1.0 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "T8Title", parent=styles["Heading1"],
        fontSize=13, spaceAfter=4,
        textColor=colors.HexColor("#0A1628"),
    )
    sub_style = ParagraphStyle(
        "T8Sub", parent=styles["Normal"],
        fontSize=8.5, textColor=colors.HexColor("#5B7083"), spaceAfter=8,
    )
    cell_style = ParagraphStyle(
        "T8Cell", parent=styles["Normal"],
        fontSize=6.5, leading=8.5, wordWrap="CJK",
    )

    story = []
    story.append(Paragraph(
        f"Top-10 Novel Potent Molecules — {method_label}", title_style))
    story.append(Paragraph(
        f"Ranked by composite score (CNNaffinity × CNNscore)  ·  "
        f"Pareto front size: {len(pareto_df)}  ·  {title_note}",
        sub_style,
    ))

    IMG_W, IMG_H = 160, 120   # pixels → scaled to cm in reportlab
    img_w_cm = 4.0 * cm
    img_h_cm = img_w_cm * (IMG_H / IMG_W)

    col_widths = [
        0.8 * cm,    # Rank
        img_w_cm,    # Drawing
        6.5 * cm,    # SMILES
        2.5 * cm,    # CNNaffinity
        2.5 * cm,    # CNNscore
        2.5 * cm,    # Vinardo
        2.5 * cm,    # Tanimoto
        2.8 * cm,    # Composite
    ]

    header = [
        Paragraph("<b>Rank</b>", cell_style),
        Paragraph("<b>Structure</b>", cell_style),
        Paragraph("<b>SMILES</b>", cell_style),
        Paragraph("<b>CNNaffinity</b><br/>(higher=better)", cell_style),
        Paragraph("<b>CNNscore</b><br/>(higher=better)", cell_style),
        Paragraph("<b>Vinardo</b><br/>(lower=better)", cell_style),
        Paragraph("<b>Tanimoto</b><br/>(lower=more novel)", cell_style),
        Paragraph("<b>Composite</b><br/>(aff×score)", cell_style),
    ]

    table_data = [header]

    def _fmt(v):
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return "—"

    for rank_i, (_, row) in enumerate(top10.iterrows(), start=1):
        png = _mol_png_bytes(str(row["smiles"]), IMG_W, IMG_H)
        if png is not None:
            img_cell = RLImage(io.BytesIO(png), width=img_w_cm, height=img_h_cm)
        else:
            img_cell = Paragraph("(cannot parse)", cell_style)

        table_data.append([
            Paragraph(str(rank_i), cell_style),
            img_cell,
            Paragraph(str(row["smiles"]), cell_style),
            Paragraph(_fmt(row["CNNaffinity"]),       cell_style),
            Paragraph(_fmt(row["CNNscore"]),          cell_style),
            Paragraph(_fmt(row["minimizedAffinity"]), cell_style),
            Paragraph(_fmt(row["tanimoto"]),          cell_style),
            Paragraph(_fmt(row["composite"]),         cell_style),
        ])

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header
        ("BACKGROUND",    (0, 0), (-1, 0), colors.white),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.HexColor("#0A1628")),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8),
        ("LINEBELOW",     (0, 0), (-1, 0), 1.2, colors.HexColor("#0A1628")),
        # Data rows
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#F0F6FB"), colors.white]),
        ("FONTSIZE",      (0, 1), (-1, -1), 6.5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 3),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        # Gold tint on rank-1
        ("BACKGROUND",    (0, 1), (-1, 1), colors.HexColor("#D4A01744")),
    ]))

    story.append(tbl)
    doc.build(story)
    print(f"  💾 Top-10 PDF saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


def run_stage8(
    csv_path:         Optional[str]   = None,
    mask_calc_dir:    Optional[str]   = None,
    stage8_dir:       Optional[str]   = None,
    y_col:            str             = "CNNaffinity",
    cnn_score_min:    float           = 0.70,
    vinardo_max:      float           = -7.0,
    ranking_methods:  Optional[set]   = None,
    tanimoto_cutoff:  float           = 0.40,
    mmr_lambda:       float           = 0.50,
    cluster_cutoff:   float           = 0.40,
    show_pareto:      bool            = False,
) -> Dict[str, object]:
    """
    Run Stage-8 analysis.

    Parameters
    ----------
    ranking_methods : set of method keys — any subset of
                      {"idea1", "idea3", "idea4", "idea5"}.
                      Default: {"idea1"}.
    tanimoto_cutoff : Tanimoto novelty cutoff for Idea 1 (default 0.40).
    mmr_lambda      : λ for MMR Idea 4 (default 0.50).
    cluster_cutoff  : Butina cutoff for Idea 5 (default 0.40).
    show_pareto     : overlay Pareto-front on scatter plots (default False).

    Returns dict with keys:
      df               : full annotated DataFrame
      pareto_df        : Pareto-front subset
      filtered_df      : hierarchical-filter subset
      scatter_paths    : list of PNG paths written
      pdf_paths        : dict {method_key → pdf_path}
    """
    csv_path        = csv_path        or config.STAGE8_INPUT_CSV
    mask_calc_dir   = mask_calc_dir   or config.MASK_CALC_OUTDIR
    stage8_dir      = stage8_dir      or config.STAGE8_DIR
    ranking_methods = ranking_methods or {"idea1"}

    os.makedirs(stage8_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    original_smiles_map = load_original_smiles(mask_calc_dir)
    print(f"  Original SMILES loaded: {len(original_smiles_map)} ligand(s)")

    df = load_docking_data(csv_path, original_smiles_map)
    print(f"  Docked molecules (pose-1): {len(df)}")
    print(f"  Molecules with Tanimoto computed: "
          f"{df['tanimoto'].notna().sum()}")

    # ── Pareto front (always computed — used by Idea 3 + optional overlay) ────
    pareto_df = pareto_front(df, score_col="composite")
    print(f"  Pareto front size: {len(pareto_df)} molecules")

    # ── Hierarchical filter (always computed — used by Idea 3 + CSV output) ───
    filtered_df = hierarchical_filter(df, cnn_score_min, vinardo_max)
    print(f"  Hierarchical filter survivors: {len(filtered_df)}")

    # ── Scatter plots ─────────────────────────────────────────────────────────
    scatter_paths = []
    groups = sorted(df["ligand_group"].unique())

    for lig_id in groups:
        df_grp   = df[df["ligand_group"] == lig_id]
        out_path = os.path.join(
            stage8_dir, f"scatter_{_safe(lig_id)}_{y_col}.png"
        )
        _scatter_one_group(df_grp, lig_id, y_col, pareto_df, out_path,
                           show_pareto=show_pareto)
        scatter_paths.append(out_path)

    combined_path = os.path.join(stage8_dir, f"scatter_combined_{y_col}.png")
    _scatter_combined(df, y_col, pareto_df, combined_path,
                      show_pareto=show_pareto)
    scatter_paths.append(combined_path)

    # ── Top-10 PDFs — one per selected method ────────────────────────────────
    csv_note = (os.path.basename(os.path.dirname(csv_path)) + "/" +
                os.path.basename(csv_path))
    pdf_paths: Dict[str, str] = {}

    method_configs = {
        "idea1": {
            "label": f"Idea 1 — Tanimoto < {tanimoto_cutoff}, ranked by composite",
            "fn":    lambda: top10_idea1(df, tanimoto_cutoff, cnn_score_min, vinardo_max),
            "fname": "top10_idea1_tanimoto_cutoff.pdf",
        },
        "idea3": {
            "label": (f"Idea 3 — Hierarchical filter (CNNscore>{cnn_score_min},"
                      f" Vinardo<{vinardo_max}) → Pareto"),
            "fn":    lambda: top10_idea3(df, pareto_df, cnn_score_min, vinardo_max),
            "fname": "top10_idea3_filter_pareto.pdf",
        },
        "idea4": {
            "label": f"Idea 4 — MMR (λ={mmr_lambda})",
            "fn":    lambda: top10_idea4_mmr(df, mmr_lambda),
            "fname": f"top10_idea4_mmr_lambda{mmr_lambda}.pdf",
        },
        "idea5": {
            "label": f"Idea 5 — Butina cluster-pick (cutoff={cluster_cutoff})",
            "fn":    lambda: top10_idea5_cluster(df, cluster_cutoff),
            "fname": f"top10_idea5_cluster_cutoff{cluster_cutoff}.pdf",
        },
    }

    for method_key in sorted(ranking_methods):
        cfg      = method_configs[method_key]
        top10_df = cfg["fn"]()
        if top10_df.empty:
            print(f"  ⚠️  {method_key}: no molecules selected, PDF skipped.")
            continue
        pdf_path = os.path.join(stage8_dir, cfg["fname"])
        build_top10_pdf(top10_df, pdf_path,
                        title_note=csv_note,
                        method_label=cfg["label"])
        pdf_paths[method_key] = pdf_path

    # ── Hierarchical filter summary CSV ──────────────────────────────────────
    filt_csv = os.path.join(stage8_dir, "hierarchical_filter_results.csv")
    filtered_df.to_csv(filt_csv, index=False)
    print(f"  💾 Hierarchical filter results: {filt_csv}")

    return {
        "df":            df,
        "pareto_df":     pareto_df,
        "filtered_df":   filtered_df,
        "scatter_paths": scatter_paths,
        "pdf_paths":     pdf_paths,
    }


# ════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED TEST
# ════════════════════════════════════════════════════════════════════════════

_TEST_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin       — similar to original
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "OC(=O)c1ccccc1O",                # salicylic acid
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",    # caffeine      — different scaffold
    "c1ccc2ccccc2c1",                  # naphthalene   — very different
    "CC(=O)O",                         # acetic acid
    "CCO",                             # ethanol
]
_ORIG_SMILES = "CC(=O)Oc1ccccc1C(=O)O"   # aspirin as original


def _run_test() -> bool:
    print("\n" + "=" * 60)
    print("STAGE 8 SELF-TEST  (synthetic docking data)")
    print("=" * 60)

    td = os.path.join(config.STAGE8_DIR, "test")
    if os.path.exists(td):
        shutil.rmtree(td)
        print(f"  🗑  Wiped existing test dir: {td}")
    os.makedirs(td)
    print(f"  📁 Created fresh test dir:  {td}\n")

    import random as _rand

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

    # ── Synthetic Stage-1 JSON ────────────────────────────────────────────────
    mask_dir = os.path.join(td, "stage1")
    os.makedirs(mask_dir)
    meta = {
        "smiles": _ORIG_SMILES,
        "ligand": {"resname": "EAM", "chain": "A", "resseq": 1},
        "masked_atom_indices": [0, 1],
    }
    with open(os.path.join(mask_dir, "EAM-A-1.meta.json"), "w") as f:
        json.dump(meta, f)

    # ── Synthetic docking CSV ─────────────────────────────────────────────────
    _rand.seed(42)
    rows = []
    for mol_idx, smi in enumerate(_TEST_SMILES, start=1):
        rows.append({
            "ligand_group":      "EAM-A-1",
            "mol_idx":           mol_idx,
            "smiles":            smi,
            "pose":              1,
            "CNNscore":          round(_rand.uniform(0.5, 0.95), 4),
            "CNNaffinity":       round(_rand.uniform(5.0, 10.0), 4),
            "minimizedAffinity": round(_rand.uniform(-10.0, -4.0), 4),
            "complex_pdb":       "",
        })
    csv_path = os.path.join(td, "docking_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Run Stage 8 ───────────────────────────────────────────────────────────
    result = run_stage8(
        csv_path      = csv_path,
        mask_calc_dir = mask_dir,
        stage8_dir    = os.path.join(td, "out"),
        y_col         = "CNNaffinity",
        cnn_score_min = 0.50,
        vinardo_max   = -4.0,
    )

    df        = result["df"]
    pareto_df = result["pareto_df"]
    filt_df   = result["filtered_df"]

    check(len(df) == len(_TEST_SMILES),
          f"all {len(_TEST_SMILES)} molecules loaded")
    check(df["tanimoto"].notna().sum() > 0,
          "Tanimoto values computed")
    check(df["composite"].notna().sum() > 0,
          "composite score computed")
    check(len(pareto_df) > 0,
          f"Pareto front non-empty ({len(pareto_df)} molecules)")

    # Pareto front correctness: no molecule on front should be dominated
    scores  = pareto_df["composite"].values
    novelty = pareto_df["novelty"].values
    dominated = False
    for i in range(len(pareto_df)):
        for j in range(len(pareto_df)):
            if i != j:
                if (scores[j] >= scores[i] and novelty[j] >= novelty[i] and
                        (scores[j] > scores[i] or novelty[j] > novelty[i])):
                    dominated = True
    check(not dominated, "no Pareto-front molecule is dominated by another")

    check(len(filt_df) >= 0,   "hierarchical filter ran without error")

    # Output files
    for fname in result["scatter_paths"]:
        check(os.path.exists(fname), f"scatter PNG exists: {os.path.basename(fname)}")
    check(os.path.exists(result["pdf_path"]),
          f"top-10 PDF exists: {os.path.basename(result['pdf_path'])}")
    check(os.path.exists(os.path.join(td, "out",
          "hierarchical_filter_results.csv")),
          "hierarchical_filter_results.csv written")

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
    print("STAGE 8: NOVELTY vs POTENCY ANALYSIS")
    print("=" * 60)

    print(f"""
  Reads docking_summary.csv from Stage 6 and produces:
    • Scatter plots: Tanimoto (X) vs docking score (Y)
        — one per ligand group + one combined
    • Pareto front: non-dominated molecules (best novel + potent trade-offs)
    • Hierarchical filter: CNNscore > T, Vinardo < T, sorted by CNNaffinity
    • Top-10 PDF: Pareto-optimal molecules with drawing + all scores

  config.STAGE8_INPUT_CSV = {config.STAGE8_INPUT_CSV}
  config.STAGE8_DIR       = {config.STAGE8_DIR}

  Smoke test uses synthetic data — no real docking required.
""")

    run_test = _ask_yes_no_default("Run the smoke test?", default=False)
    if run_test:
        ok = _run_test()
        sys.exit(0 if ok else 1)

    # ── CSV path ──────────────────────────────────────────────────────────────
    csv_path = _resolve_csv_path()

    # ── Y axis ────────────────────────────────────────────────────────────────
    y_col = _ask_y_score()

    # ── Pareto overlay on scatter ─────────────────────────────────────────────
    show_pareto = _ask_show_pareto()

    # ── Ranking methods ───────────────────────────────────────────────────────
    methods = _ask_ranking_methods()

    # ── Method-specific parameters ────────────────────────────────────────────
    tanimoto_cutoff = 0.40
    mmr_lambda      = 0.50
    cluster_cutoff  = 0.40
    cnn_t, vin_t    = 0.70, -7.0

    if "idea1" in methods:
        tanimoto_cutoff = _ask_tanimoto_cutoff()

    if "idea3" in methods or True:   # filter thresholds also used for CSV output
        apply_filter = _ask_yes_no_default(
            "Set hierarchical filter thresholds (CNNscore + Vinardo)?",
            default=True,
        )
        if apply_filter:
            cnn_t, vin_t = _ask_hierarchical_thresholds()

    if "idea4" in methods:
        mmr_lambda = _ask_mmr_lambda()

    if "idea5" in methods:
        cluster_cutoff = _ask_cluster_cutoff()

    # ── Run ───────────────────────────────────────────────────────────────────
    result = run_stage8(
        csv_path        = csv_path,
        y_col           = y_col,
        cnn_score_min   = cnn_t,
        vinardo_max     = vin_t,
        ranking_methods = methods,
        tanimoto_cutoff = tanimoto_cutoff,
        mmr_lambda      = mmr_lambda,
        cluster_cutoff  = cluster_cutoff,
        show_pareto     = show_pareto,
    )

    print("\n" + "=" * 60)
    print("✅ Stage 8 complete.")
    print(f"   Output dir      : {config.STAGE8_DIR}")
    print(f"   Pareto front    : {len(result['pareto_df'])} molecules")
    print(f"   Filter survivors: {len(result['filtered_df'])} molecules")
    print(f"   Scatter plots   : {len(result['scatter_paths'])} PNG(s)")
    for k, p in result['pdf_paths'].items():
        print(f"   PDF [{k}]      : {p}")


if __name__ == "__main__":
    main()

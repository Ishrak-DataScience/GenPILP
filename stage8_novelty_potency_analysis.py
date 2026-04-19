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
    df = pd.read_csv(csv_path)
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
#  SCATTER PLOTS
# ════════════════════════════════════════════════════════════════════════════

def _scatter_one_group(
    df_group:   pd.DataFrame,
    lig_id:     str,
    y_col:      str,
    pareto_df:  pd.DataFrame,
    out_path:   str,
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

    # Pareto front subset for this group
    pf_group = pareto_df[pareto_df["ligand_group"] == lig_id]
    if len(pf_group) > 0:
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
        if len(pf_grp) > 0:
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

    star_patch = mpatches.Patch(
        facecolor="white", edgecolor="black",
        label="★ Pareto front molecules",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [star_patch],
              fontsize=8.5, framealpha=0.85, ncol=2)

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
    pareto_df:  pd.DataFrame,
    out_path:   str,
    title_note: str = "",
) -> None:
    """
    Build a PDF report of the top-10 Pareto-front molecules (by composite
    score descending), one row per molecule with drawing + all scores.

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
        "Top-10 Novel Potent Molecules — Pareto Front", title_style))
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
    csv_path:       Optional[str]   = None,
    mask_calc_dir:  Optional[str]   = None,
    stage8_dir:     Optional[str]   = None,
    y_col:          str             = "CNNaffinity",
    cnn_score_min:  float           = 0.70,
    vinardo_max:    float           = -7.0,
) -> Dict[str, object]:
    """
    Run Stage-8 analysis.

    Returns dict with keys:
      df               : full annotated DataFrame
      pareto_df        : Pareto-front subset
      filtered_df      : hierarchical-filter subset
      scatter_paths    : list of PNG paths written
      pdf_path         : path to top-10 PDF
    """
    csv_path      = csv_path      or config.STAGE8_INPUT_CSV
    mask_calc_dir = mask_calc_dir or config.MASK_CALC_OUTDIR
    stage8_dir    = stage8_dir    or config.STAGE8_DIR

    os.makedirs(stage8_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    original_smiles_map = load_original_smiles(mask_calc_dir)
    print(f"  Original SMILES loaded: {len(original_smiles_map)} ligand(s)")

    df = load_docking_data(csv_path, original_smiles_map)
    print(f"  Docked molecules (pose-1): {len(df)}")
    print(f"  Molecules with Tanimoto computed: "
          f"{df['tanimoto'].notna().sum()}")

    # ── Pareto front ──────────────────────────────────────────────────────────
    pareto_df = pareto_front(df, score_col="composite")
    print(f"  Pareto front size: {len(pareto_df)} molecules")

    # ── Hierarchical filter ───────────────────────────────────────────────────
    filtered_df = hierarchical_filter(df, cnn_score_min, vinardo_max)
    print(f"  Hierarchical filter survivors: {len(filtered_df)}")

    # ── Scatter plots ─────────────────────────────────────────────────────────
    scatter_paths = []
    groups = sorted(df["ligand_group"].unique())

    # Per-group scatter
    for lig_id in groups:
        df_grp   = df[df["ligand_group"] == lig_id]
        out_path = os.path.join(
            stage8_dir, f"scatter_{_safe(lig_id)}_{y_col}.png"
        )
        _scatter_one_group(df_grp, lig_id, y_col, pareto_df, out_path)
        scatter_paths.append(out_path)

    # Combined scatter
    combined_path = os.path.join(stage8_dir, f"scatter_combined_{y_col}.png")
    _scatter_combined(df, y_col, pareto_df, combined_path)
    scatter_paths.append(combined_path)

    # ── Top-10 PDF ────────────────────────────────────────────────────────────
    pdf_path = os.path.join(stage8_dir, "top10_pareto_molecules.pdf")
    csv_note = os.path.basename(os.path.dirname(csv_path)) + "/" + \
               os.path.basename(csv_path)
    build_top10_pdf(pareto_df, pdf_path, title_note=csv_note)

    # ── Hierarchical filter summary ───────────────────────────────────────────
    filt_csv = os.path.join(stage8_dir, "hierarchical_filter_results.csv")
    filtered_df.to_csv(filt_csv, index=False)
    print(f"  💾 Hierarchical filter results: {filt_csv}")

    return {
        "df":            df,
        "pareto_df":     pareto_df,
        "filtered_df":   filtered_df,
        "scatter_paths": scatter_paths,
        "pdf_path":      pdf_path,
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

    # ── Hierarchical filter ───────────────────────────────────────────────────
    apply_filter = _ask_yes_no_default(
        "Apply hierarchical filter (CNNscore + Vinardo thresholds)?",
        default=True,
    )
    if apply_filter:
        cnn_t, vin_t = _ask_hierarchical_thresholds()
    else:
        cnn_t, vin_t = 0.0, 0.0   # no-op thresholds

    # ── Run ───────────────────────────────────────────────────────────────────
    result = run_stage8(
        csv_path      = csv_path,
        y_col         = y_col,
        cnn_score_min = cnn_t,
        vinardo_max   = vin_t,
    )

    print("\n" + "=" * 60)
    print("✅ Stage 8 complete.")
    print(f"   Output dir      : {config.STAGE8_DIR}")
    print(f"   Pareto front    : {len(result['pareto_df'])} molecules")
    print(f"   Filter survivors: {len(result['filtered_df'])} molecules")
    print(f"   Scatter plots   : {len(result['scatter_paths'])} PNG(s)")
    print(f"   Top-10 PDF      : {result['pdf_path']}")


if __name__ == "__main__":
    main()

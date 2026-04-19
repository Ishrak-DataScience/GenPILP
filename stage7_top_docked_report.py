# -*- coding: utf-8 -*-
"""
stage7_top_docked_report.py
===========================
Stage 7 of the pipeline — top-docked molecule PDF report.

For each original ligand group (e.g. EAM-A-1), produces one page in a
single PDF containing:

  1. Page title  — ligand group name + ranking criterion
  2. Molecule row image:
       [ Original ligand ] [ Rank-1 ] [ Rank-2 ] [ Rank-3 ] [ Rank-4 ] [ Rank-5 ]
       The original molecule cell is bordered:
         GREEN  (0,180,0)   — if the original was reproduced among the top-5
         ORANGE (255,140,0) — if not (always the case unless ChemBERTa exactly
                              regenerated the parent ligand at a top-5 position)
  3. Scores table — one row per top-5 molecule:
       Rank | SMILES (full) | CNNaffinity | CNNscore | minimizedAffinity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1.  Input CSV is config.STAGE6_DIR/docking_summary.csv (Stage 6 output).
     Original ligand SMILES are read from Stage-1 .meta.json files in
     config.MASK_CALC_OUTDIR.

A2.  Only pose 1 per molecule is used for ranking (GNINA top-ranked pose).

A3.  Ranking criterion is asked at runtime (default CNNaffinity).
     CNNaffinity and CNNscore: higher = better.
     minimizedAffinity (Vinardo): lower = better — the list is sorted
     ascending when this criterion is selected.

A4.  If a ligand group has fewer than 5 docked molecules, all available
     molecules are shown with a note.

A5.  The original molecule cell is always shown first (leftmost) in the
     molecule row. Border colour follows Stage 3 convention:
       GREEN  — original SMILES reproduced among the top-5
       ORANGE — original not in top-5 (most common case)

A6.  The full SMILES string is shown in the table. Long strings wrap
     within the cell.

A7.  Output: config.STAGE7_DIR/top_docked_report.pdf
     One additional PNG per ligand group saved alongside the PDF for
     easy inspection without opening the full PDF:
       config.STAGE7_DIR/<ligand_id>_top5_molecules.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install reportlab          # if not already installed
  python stage7_top_docked_report.py

Must be run AFTER stage6_docking.py.

HOW TO TEST (no real docking data required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Answer "yes" to the smoke-test prompt.
  Writes to config.STAGE7_DIR/test/ (wiped each run).
"""

import glob
import io
import json
import os
import shutil
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

import config

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

SCORE_COLS = ["CNNaffinity", "CNNscore", "minimizedAffinity"]

SCORE_LABELS = {
    "CNNaffinity":       "CNN Affinity  (higher = better)",
    "CNNscore":          "CNN Score  (higher = better)",
    "minimizedAffinity": "Vinardo  (lower = better)",
}

# For minimizedAffinity sorting is ascending; all others descending
LOWER_IS_BETTER = {"minimizedAffinity"}

# Molecule image size (pixels) per cell in the row image
MOL_IMG_W = 280
MOL_IMG_H = 220

# Border colours — same convention as Stage 3
BORDER_GREEN  = (0,   180,   0)   # original reproduced in top-5
BORDER_ORANGE = (255, 140,   0)   # original not in top-5
BORDER_WIDTH  = 8                  # pixels

TOP_N = 5


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
        print("    Please type 'yes'/'y' or 'no'/'n', or press Enter for default.")


def _ask_ranking_criterion() -> str:
    """Ask the user which score to use for ranking. Returns column name."""
    print("""
  Ranking criterion — select the score used to determine the top-5 molecules:

    1. CNNaffinity       — CNN-predicted -log Kd/Ki     (higher = better)  [default]
    2. CNNscore          — CNN binding probability 0-1  (higher = better)
    3. minimizedAffinity — Vinardo score (kcal/mol)     (lower  = better)
""")
    mapping = {"1": "CNNaffinity", "2": "CNNscore", "3": "minimizedAffinity",
               "cnnaffinity": "CNNaffinity", "cnnscore": "CNNscore",
               "minimizedaffinity": "minimizedAffinity",
               "vinardo": "minimizedAffinity"}
    while True:
        raw = input("  Enter 1 / 2 / 3 or score name [1]: ").strip().lower()
        if raw == "":
            return "CNNaffinity"
        if raw in mapping:
            return mapping[raw]
        print("    Please enter 1, 2, or 3.")


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_original_smiles(mask_calc_dir: str) -> Dict[str, str]:
    """Load {ligand_key → original SMILES} from Stage-1 .meta.json files."""
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
                result[key] = Chem.MolToSmiles(mol)   # canonical form
        except Exception as e:
            print(f"  ⚠️  Could not read {os.path.basename(path)}: {e}")
    return result


def load_and_rank(
    csv_path: str,
    criterion: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load docking_summary.csv, keep pose-1 rows only, coerce score columns,
    sort by criterion, return {ligand_group → ranked DataFrame}.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"docking_summary.csv not found at {csv_path}.\n"
            "Run stage6_docking.py first."
        )

    df = pd.read_csv(csv_path)
    for c in SCORE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Pose 1 only (Assumption A2)
    df = df[df["pose"] == 1].copy().reset_index(drop=True)

    ascending = criterion in LOWER_IS_BETTER
    df = df.sort_values(criterion, ascending=ascending).reset_index(drop=True)

    result: Dict[str, pd.DataFrame] = {}
    for grp, sub in df.groupby("ligand_group"):
        result[str(grp)] = sub.reset_index(drop=True)

    return result


# ════════════════════════════════════════════════════════════════════════════
#  MOLECULE ROW IMAGE
# ════════════════════════════════════════════════════════════════════════════

def _mol_to_png_bytes(smi: str, w: int, h: int) -> Optional[bytes]:
    """Render one molecule to PNG bytes via RDKit, handling return-type variation."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(w, h))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _draw_border(img_bytes: bytes, color: tuple, bw: int,
                 w: int, h: int) -> bytes:
    """Draw a thick coloured border on a molecule PNG (PIL)."""
    from PIL import Image as PilImage, ImageDraw as PilDraw
    img  = PilImage.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = PilDraw.Draw(img)
    for i in range(bw):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _add_label(img_bytes: bytes, label: str, w: int, h: int) -> bytes:
    """Add a centred label banner at the top of a molecule PNG."""
    from PIL import Image as PilImage, ImageDraw as PilDraw, ImageFont
    banner_h = 22
    img = PilImage.open(io.BytesIO(img_bytes)).convert("RGB")
    # Create new image with banner
    new_img = PilImage.new("RGB", (w, h + banner_h), (240, 240, 240))
    new_img.paste(img, (0, banner_h))
    draw = PilDraw.Draw(new_img)
    draw.rectangle([0, 0, w, banner_h], fill=(220, 220, 220))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    # Centre text
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        tx = max(0, (w - (bbox[2] - bbox[0])) // 2)
    except AttributeError:
        tx = 4
    draw.text((tx, 3), label, fill=(50, 50, 50), font=font)
    buf = io.BytesIO()
    new_img.save(buf, format="PNG")
    return buf.getvalue()


def load_strategy_sets(
    pred_dir: str,
    lig_id: str,
) -> Tuple[set, set]:
    """
    Read all ia_mask*.txt and rand_mask*.txt for lig_id and return
    (ia_canon_set, rand_canon_set) of canonical SMILES strings.

    Uses the same file structure as Stage 2 / Stage 3.
    """
    import glob as _glob

    # Stage 2 writes folders using safe_filename (same rule as _safe).
    # Try exact lig_id first, then the safe version as a fallback.
    lig_dir      = os.path.join(pred_dir, lig_id)
    tried_paths  = [lig_dir]
    if not os.path.isdir(lig_dir):
        safe_id  = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in lig_id)
        lig_dir  = os.path.join(pred_dir, safe_id)
        tried_paths.append(lig_dir)

    print(f"    📂 Strategy lookup for '{lig_id}':")
    print(f"       Base PRED_DIR  : {pred_dir}")
    for p in tried_paths:
        exists = "✅ found" if os.path.isdir(p) else "❌ not found"
        print(f"       Folder tried  : {p}  [{exists}]")

    if not os.path.isdir(lig_dir):
        ia_files   = []
        rand_files = []
    else:
        ia_files   = sorted(_glob.glob(os.path.join(lig_dir, "ia_mask*.txt")))
        rand_files = sorted(_glob.glob(os.path.join(lig_dir, "rand_mask*.txt")))

    print(f"       ia_mask*.txt  : {len(ia_files)} file(s) found")
    print(f"       rand_mask*.txt: {len(rand_files)} file(s) found")

    if not ia_files and not rand_files:
        print(f"       ⚠️  No prediction files — Strategy column will show '—'.")
        print(f"          Dependency: run stage2_molecule_generation.py first")
        print(f"          and verify config.PRED_DIR = {pred_dir}")
        return set(), set()

    ia_set:   set = set()
    rand_set: set = set()

    for path in ia_files:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    smi = line.strip()
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        ia_set.add(Chem.MolToSmiles(mol))
        except Exception:
            pass

    for path in rand_files:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    smi = line.strip()
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        rand_set.add(Chem.MolToSmiles(mol))
        except Exception:
            pass

    print(f"       Loaded: {len(ia_set)} unique IA SMILES, "
          f"{len(rand_set)} unique random SMILES")
    return ia_set, rand_set


def _strategy_tag(canon_smi: Optional[str],
                  ia_set: set, rand_set: set) -> str:
    """
    Return the strategy tag for one SMILES:
      "IA"  — in ia_set only
      "R"   — in rand_set only
      "B"   — in both
      ""    — not found in either (e.g. if pred files are missing)
    """
    if canon_smi is None:
        return ""
    in_ia   = canon_smi in ia_set
    in_rand = canon_smi in rand_set
    if in_ia and in_rand:
        return "B"
    if in_ia:
        return "IA"
    if in_rand:
        return "R"
    return ""


def build_molecule_row_image(
    original_smi: Optional[str],
    top_smiles: List[str],
    in_top5: bool,
    strategy_tags: Optional[List[str]] = None,
) -> bytes:
    """
    Build a horizontal strip of molecule images:
      [ Original | Rank1(IA) | Rank2(R) | Rank3(B) | Rank4(IA) | Rank5(R) ]

    Returns PNG bytes of the full strip.

    Parameters
    ----------
    original_smi  : canonical SMILES of original ligand (may be None)
    top_smiles    : list of SMILES in rank order (up to TOP_N)
    in_top5       : True if original was found among top_smiles
    strategy_tags : list of tags parallel to top_smiles —
                    "IA", "R", "B", or "" for unknown.
                    If None or shorter than top_smiles, missing tags shown as "".
    """
    from PIL import Image as PilImage

    banner_h = 22
    cell_h   = MOL_IMG_H + banner_h

    cells_bytes: List[bytes] = []

    # ── Original molecule cell ────────────────────────────────────────────────
    if original_smi is not None:
        orig_png = _mol_to_png_bytes(original_smi, MOL_IMG_W, MOL_IMG_H)
        if orig_png is None:
            orig_png = _blank_cell(MOL_IMG_W, MOL_IMG_H, "Cannot parse")
        border_col = BORDER_GREEN if in_top5 else BORDER_ORANGE
        orig_png   = _draw_border(orig_png, border_col, BORDER_WIDTH,
                                  MOL_IMG_W, MOL_IMG_H)
        label = "Original ★" if in_top5 else "Original ⊕"
        orig_png = _add_label(orig_png, label, MOL_IMG_W, MOL_IMG_H)
    else:
        orig_png = _blank_cell(MOL_IMG_W, cell_h, "No original SMILES")
        # no banner needed — blank cell already full height
        cell_h_orig = cell_h
    cells_bytes.append(orig_png)

    # ── Top-N docked molecule cells ───────────────────────────────────────────
    for rank, smi in enumerate(top_smiles, start=1):
        png = _mol_to_png_bytes(smi, MOL_IMG_W, MOL_IMG_H)
        if png is None:
            png = _blank_cell(MOL_IMG_W, MOL_IMG_H, "Cannot parse")
        tag = ""
        if strategy_tags is not None and rank - 1 < len(strategy_tags):
            tag = strategy_tags[rank - 1]
        lbl = f"Rank {rank} ({tag})" if tag else f"Rank {rank}"
        png = _add_label(png, lbl, MOL_IMG_W, MOL_IMG_H)
        cells_bytes.append(png)

    # ── Stitch horizontally ───────────────────────────────────────────────────
    imgs = [PilImage.open(io.BytesIO(b)).convert("RGB") for b in cells_bytes]
    total_w = sum(im.width for im in imgs)
    max_h   = max(im.height for im in imgs)

    strip = PilImage.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in imgs:
        strip.paste(im, (x, max_h - im.height))   # bottom-align
        x += im.width

    buf = io.BytesIO()
    strip.save(buf, format="PNG")
    return buf.getvalue()


def _blank_cell(w: int, h: int, msg: str = "") -> bytes:
    """Return a plain grey PNG cell with optional centred message."""
    from PIL import Image as PilImage, ImageDraw as PilDraw
    img  = PilImage.new("RGB", (w, h), (220, 220, 220))
    if msg:
        draw = PilDraw.Draw(img)
        draw.text((4, h // 2), msg, fill=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
#  PDF GENERATION  (reportlab Platypus)
# ════════════════════════════════════════════════════════════════════════════

def _build_pdf(
    ligand_groups: Dict[str, pd.DataFrame],
    original_smiles_map: Dict[str, str],
    criterion: str,
    pdf_path: str,
    png_dir: str,
    pred_dir: Optional[str] = None,
) -> None:
    """
    Build the multi-page PDF.  One page per ligand group.
    Also saves a standalone PNG per group to png_dir.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    page_w, page_h = landscape(A4)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(A4),
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "GroupTitle",
        parent=styles["Heading1"],
        fontSize=15,
        spaceAfter=6,
        textColor=colors.HexColor("#0A1628"),
    )
    sub_style = ParagraphStyle(
        "SubTitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#5B7083"),
        spaceAfter=10,
    )
    cell_style = ParagraphStyle(
        "CellWrap",
        parent=styles["Normal"],
        fontSize=7,
        leading=9,
        wordWrap="CJK",
    )

    story = []
    groups_sorted = sorted(ligand_groups.keys())

    for g_idx, lig_id in enumerate(groups_sorted):
        ranked_df = ligand_groups[lig_id]
        top_df    = ranked_df.head(TOP_N).copy()
        top_n_actual = len(top_df)

        orig_smi = original_smiles_map.get(lig_id)
        if orig_smi is not None:
            orig_canon = Chem.MolToSmiles(Chem.MolFromSmiles(orig_smi)) if Chem.MolFromSmiles(orig_smi) else None
        else:
            orig_canon = None

        top_smiles = top_df["smiles"].tolist()
        top_canons = []
        for s in top_smiles:
            m = Chem.MolFromSmiles(str(s))
            top_canons.append(Chem.MolToSmiles(m) if m else None)

        in_top5 = orig_canon is not None and orig_canon in top_canons

        # ── Page title ────────────────────────────────────────────────────────
        story.append(Paragraph(f"Ligand Group: {lig_id}", title_style))
        direction = "lower = better" if criterion in LOWER_IS_BETTER else "higher = better"
        story.append(Paragraph(
            f"Top {top_n_actual} docked molecules ranked by "
            f"<b>{criterion}</b> ({direction})   |   "
            f"Original ligand: {'reproduced in top-5 ★' if in_top5 else 'not in top-5 ⊕'}",
            sub_style,
        ))

        # ── Strategy tags for top-5 molecules ────────────────────────────────
        strategy_tags: List[str] = []
        if pred_dir is not None:
            ia_set, rand_set = load_strategy_sets(pred_dir, lig_id)
            if not ia_set and not rand_set:
                print(f"  ⚠️  {lig_id}: no Stage-2 prediction files found in "
                      f"{pred_dir} — Strategy column will show '—'.")
                print(f"       Dependency: run stage2_molecule_generation.py "
                      f"first and ensure PRED_DIR is set correctly in config.py.")
            else:
                print(f"  ℹ️  {lig_id}: {len(ia_set)} IA + {len(rand_set)} random "
                      f"SMILES loaded for strategy tagging.")
            for smi in top_smiles:
                mol = Chem.MolFromSmiles(str(smi))
                canon = Chem.MolToSmiles(mol) if mol else None
                strategy_tags.append(_strategy_tag(canon, ia_set, rand_set))
        else:
            print(f"  ⚠️  {lig_id}: pred_dir not set — Strategy column will show '—'.")

        # ── Molecule row image ────────────────────────────────────────────────
        row_png = build_molecule_row_image(
            orig_smi, top_smiles, in_top5,
            strategy_tags=strategy_tags or None,
        )

        # Save standalone PNG
        png_path = os.path.join(png_dir, f"{_safe(lig_id)}_top5_molecules.png")
        with open(png_path, "wb") as f:
            f.write(row_png)

        # Scale image to fit page width
        available_w = page_w - 3.0 * cm
        from PIL import Image as PilImage
        pil_img  = PilImage.open(io.BytesIO(row_png))
        img_w_px, img_h_px = pil_img.size
        scale    = available_w / img_w_px
        rl_img_w = available_w
        rl_img_h = img_h_px * scale

        story.append(RLImage(io.BytesIO(row_png), width=rl_img_w, height=rl_img_h))
        story.append(Spacer(1, 0.4 * cm))

        # ── Scores table ──────────────────────────────────────────────────────
        col_widths = [1.0 * cm, 8.0 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 2.0 * cm]

        header = [
            Paragraph("<b>Rank</b>", cell_style),
            Paragraph("<b>SMILES</b>", cell_style),
            Paragraph("<b>CNNaffinity</b><br/>(higher=better)", cell_style),
            Paragraph("<b>CNNscore</b><br/>(higher=better)", cell_style),
            Paragraph("<b>Vinardo</b><br/>(lower=better)", cell_style),
            Paragraph("<b>Strategy</b><br/>(IA / R / B)", cell_style),
        ]
        table_data = [header]

        for rank_i, row in enumerate(top_df.itertuples(), start=1):
            cnn_aff  = row.CNNaffinity
            cnn_sc   = row.CNNscore
            vinardo  = row.minimizedAffinity

            def fmt(v):
                try:
                    return f"{float(v):.4f}"
                except (TypeError, ValueError):
                    return str(v)

            stag = strategy_tags[rank_i - 1] if rank_i - 1 < len(strategy_tags) else ""
            stag_full = {"IA": "IA (interaction-aware)",
                         "R":  "R (random)",
                         "B":  "B (both)"}.get(stag, stag or "—")
            table_data.append([
                Paragraph(str(rank_i), cell_style),
                Paragraph(str(row.smiles), cell_style),
                Paragraph(fmt(cnn_aff),  cell_style),
                Paragraph(fmt(cnn_sc),   cell_style),
                Paragraph(fmt(vinardo),  cell_style),
                Paragraph(stag_full,     cell_style),
            ])

        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)

        # Header: white background, dark text, bold, bottom border
        tbl_style = TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#0A1628")),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 8),
            # Separator line under header to distinguish it from data rows
            ("LINEBELOW",   (0, 0), (-1, 0), 1.2, colors.HexColor("#0A1628")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#F0F6FB"), colors.white]),
            ("FONTSIZE",    (0, 1), (-1, -1), 7.5),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ])

        # Highlight the row of the top-ranked molecule
        if criterion in LOWER_IS_BETTER:
            tbl_style.add("BACKGROUND", (0, 1), (-1, 1),
                          colors.HexColor("#D4A01744"))
        else:
            tbl_style.add("BACKGROUND", (0, 1), (-1, 1),
                          colors.HexColor("#D4A01744"))   # gold tint for rank 1

        tbl.setStyle(tbl_style)
        story.append(tbl)

        if g_idx < len(groups_sorted) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"  💾 PDF saved: {pdf_path}")


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_stage7(
    csv_path:       Optional[str] = None,
    mask_calc_dir:  Optional[str] = None,
    stage7_dir:     Optional[str] = None,
    criterion:      str = "CNNaffinity",
    top_n:          int = TOP_N,
) -> str:
    """
    Build the top-docked PDF report.

    Returns the path to the saved PDF.
    """
    csv_path      = csv_path      or os.path.join(config.STAGE6_DIR,
                                                   "docking_summary.csv")
    mask_calc_dir = mask_calc_dir or config.MASK_CALC_OUTDIR
    stage7_dir    = stage7_dir    or config.STAGE7_DIR

    os.makedirs(stage7_dir, exist_ok=True)

    print(f"\n  Loading docking results from:\n    {csv_path}")
    ligand_groups       = load_and_rank(csv_path, criterion)
    original_smiles_map = load_original_smiles(mask_calc_dir)

    print(f"  Ligand groups found : {len(ligand_groups)}")
    print(f"  Original SMILES     : {len(original_smiles_map)}")
    print(f"  Ranking criterion   : {criterion}  ({SCORE_LABELS[criterion]})")

    for lig_id, df in sorted(ligand_groups.items()):
        n = min(top_n, len(df))
        print(f"  {lig_id}: {len(df)} molecules → showing top {n}")

    pdf_path = os.path.join(stage7_dir, "top_docked_report.pdf")
    _build_pdf(
        ligand_groups       = ligand_groups,
        original_smiles_map = original_smiles_map,
        criterion           = criterion,
        pdf_path            = pdf_path,
        png_dir             = stage7_dir,
        pred_dir            = config.PRED_DIR,
    )

    return pdf_path


# ════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED TEST
# ════════════════════════════════════════════════════════════════════════════

_TEST_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "OC(=O)c1ccccc1O",                # salicylic acid
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",    # caffeine
    "c1ccc2ccccc2c1",                  # naphthalene
]

_ORIGINAL_SMILES = "CC(=O)Oc1ccccc1C(=O)O"   # aspirin


def _run_test() -> bool:
    import random

    print("\n" + "=" * 60)
    print("STAGE 7 SELF-TEST  (synthetic docking data)")
    print("=" * 60)

    td = os.path.join(config.STAGE7_DIR, "test")
    if os.path.exists(td):
        shutil.rmtree(td)
        print(f"  🗑  Wiped existing test dir: {td}")
    os.makedirs(td)
    print(f"  📁 Created fresh test dir:  {td}\n")

    # ── Synthetic Stage-1 JSON ────────────────────────────────────────────────
    pred_dir_t    = os.path.join(td, "preds")
    mask_calc_dir = os.path.join(td, "stage1")
    os.makedirs(mask_calc_dir)
    for resname, chain, resseq, smi in [
        ("EAM", "A", 1, _ORIGINAL_SMILES),
        ("JQ1", "A", 201, "Cc1sc2c(c1C)C(=N[CH](CC(=O)OC(C)(C)C)c3[nH]nc(C)[n+]23)c4ccc(Cl)cc4"),
    ]:
        key  = f"{resname}-{chain}-{resseq}"
        meta = {
            "smiles": smi,
            "ligand": {"resname": resname, "chain": chain, "resseq": resseq},
            "masked_atom_indices": [0, 1],
        }
        with open(os.path.join(mask_calc_dir, f"{key}.meta.json"), "w") as f:
            json.dump(meta, f)

    # ── Synthetic docking_summary.csv ────────────────────────────────────────
    random.seed(42)
    rows = []
    for lig_id, orig in [("EAM-A-1", _ORIGINAL_SMILES),
                          ("JQ1-A-201", _TEST_SMILES[1])]:
        for mol_idx, smi in enumerate(_TEST_SMILES, start=1):
            rows.append({
                "ligand_group":      lig_id,
                "mol_idx":           mol_idx,
                "smiles":            smi,
                "pose":              1,
                "CNNscore":          round(random.uniform(0.4, 0.95), 4),
                "CNNaffinity":       round(random.uniform(5.0, 10.0), 4),
                "minimizedAffinity": round(random.uniform(-10.0, -5.0), 4),
                "complex_pdb":       "",
            })

    csv_path = os.path.join(td, "docking_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Run Stage 7 ───────────────────────────────────────────────────────────
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

    # Write synthetic ia_mask / rand_mask txt files so strategy tags are tested
    for lig_id_t, smiles_set in [("EAM-A-1",   _TEST_SMILES[:3]),
                                  ("JQ1-A-201", _TEST_SMILES[3:])]:
        lig_pred = os.path.join(pred_dir_t, _safe(lig_id_t))
        os.makedirs(lig_pred, exist_ok=True)
        with open(os.path.join(lig_pred, "ia_mask001.txt"), "w") as f:
            f.write("\n".join(smiles_set[:2]) + "\n")
        with open(os.path.join(lig_pred, "rand_mask001.txt"), "w") as f:
            f.write("\n".join(smiles_set[1:3]) + "\n")  # overlap on index 1 → B

    pdf_path = run_stage7(
        csv_path      = csv_path,
        mask_calc_dir = mask_calc_dir,
        stage7_dir    = td,
        criterion     = "CNNaffinity",
    )

    check(os.path.exists(pdf_path),
          f"PDF created at {pdf_path}")
    check(os.path.getsize(pdf_path) > 5000,
          f"PDF is non-trivially sized ({os.path.getsize(pdf_path)} bytes)")

    # Check standalone PNGs
    for lig_id in ["EAM-A-1", "JQ1-A-201"]:
        png = os.path.join(td, f"{_safe(lig_id)}_top5_molecules.png")
        check(os.path.exists(png), f"{lig_id} standalone PNG exists")

    # Check CSV loads and ranking works for all criteria
    for crit in SCORE_COLS:
        try:
            groups = load_and_rank(csv_path, crit)
            check(len(groups) == 2, f"load_and_rank with {crit} returns 2 groups")
        except Exception as e:
            check(False, f"load_and_rank with {crit}: {e}")

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
    print("STAGE 7: TOP-DOCKED MOLECULE PDF REPORT")
    print("=" * 60)

    print(f"""
  This stage reads docking_summary.csv (Stage 6 output) and produces a
  single PDF with one page per ligand group showing:
    • A molecule row: original ligand + top-5 docked molecules (with borders)
    • A scores table: full SMILES, CNNaffinity, CNNscore, Vinardo

  Input CSV   : {os.path.join(config.STAGE6_DIR, 'docking_summary.csv')}
  Output PDF  : {os.path.join(config.STAGE7_DIR, 'top_docked_report.pdf')}
  Standalone PNGs: {config.STAGE7_DIR}

  Before running, you can try a smoke test with synthetic docking data.
  Writes to: {os.path.join(config.STAGE7_DIR, 'test')}
  (wiped clean at the start of every test run).
""")

    run_test = _ask_yes_no_default("Run the smoke test?", default=False)
    if run_test:
        ok = _run_test()
        sys.exit(0 if ok else 1)

    criterion = _ask_ranking_criterion()

    pdf_path = run_stage7(criterion=criterion)

    print("\n" + "=" * 60)
    print("✅ Stage 7 complete.")
    print(f"   PDF saved to    : {pdf_path}")
    print(f"   PNGs saved to   : {config.STAGE7_DIR}")


if __name__ == "__main__":
    main()

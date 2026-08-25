# -*- coding: utf-8 -*-
"""
stage3_analysis.py
==================
Stage 3 of the pipeline — post-generation analysis.

Reads Stage-2 prediction txt files and Stage-1 JSON files, then for each
ligand produces:

  Aggregated level  (always):
    • molecule_grid.png        — all unique valid SMILES drawn together
    • histogram_pairwise.png   — Tanimoto similarity of every pair within the set
    • histogram_vs_original.png — Tanimoto similarity of each molecule vs the
                                  original (unmasked) ligand SMILES

  Per-mask-count level  (if user requests):
    Same three files, one set per mask_count step (1 … N).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1.  Similarity metric: Tanimoto on Morgan fingerprints (radius=2, 2048 bits).

A2.  "Pairwise histogram" requires at least 2 molecules in the pool.
     If fewer than 2 are present, the pairwise histogram is skipped and
     a warning is printed.

A3.  Molecule grid is capped at MAX_GRID_MOLS (default 50) to keep image
     size reasonable. If the pool exceeds this cap, the first 50 are drawn
     and a warning is printed. Similarity analysis always uses the full pool.

A4.  The original ligand SMILES is read from the Stage-1 .meta.json files
     in config.MASK_CALC_OUTDIR, matched to each ligand by the key
     "{resname}-{chain}-{resseq}". If no Stage-1 JSON is found for a ligand
     the "vs original" histogram is skipped with a warning.

A5.  IA and random SMILES are pooled per group. Deduplication (Option A)
     means each unique SMILES string is counted once regardless of which
     strategy produced it.

A6.  Ligand folders under config.PRED_DIR are discovered by listing
     sub-directories. Each sub-directory name is treated as a ligand ID.
     Files matching ia_mask*.txt and rand_mask*.txt are read.

A7.  Each txt file produced by Stage 2 contains one SMILES per line
     (plus an optional header line starting with a letter that is not
     a valid SMILES start — skipped automatically by RDKit validation).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python stage3_analysis.py

Must be run AFTER stage2_molecule_generation.py.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python stage3_analysis.py --test

Creates synthetic Stage-2 prediction files for two ligands (aspirin,
caffeine) with hard-coded SMILES, runs the full analysis, and checks that
all expected output files were produced.
"""

import glob
import json
import os
import re
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in Colab and headless environments
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity
import os
import io
import config




# ════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PROMPTS
# ════════════════════════════════════════════════════════════════════════════

def _ask_yes_no_default(question: str, default: bool) -> bool:
    """
    Prompt the user with a yes/no question.
    Pressing Enter without typing accepts `default`.
    Loops until a valid answer is given.
    """
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"  {question} {hint}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("yes", "y"):
            return True
        if raw in ("no", "n"):
            return False
        print("    Please type 'yes' / 'y' or 'no' / 'n', or press Enter for the default.")


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
        print("    Please type \'ia\', \'rand\', or \'both\'.")


def ask_runtime_options() -> Tuple[bool, bool, str, str]:
    """
    Ask all Stage-3 design questions at runtime.
    Returns (per_mask_count_outputs, deduplicate, stage_choice, pool_choice).
    """
    print("\n" + "─" * 60)
    print("STAGE 3 OPTIONS")
    print("─" * 60)

    print("""
  Option 1 — Per-mask-count outputs:
    In addition to one aggregated analysis per ligand (all mask counts
    pooled), Stage 3 can also produce a separate grid + histograms for
    every individual mask-count step (mask_count = 1, 2, … N).
    This gives fine-grained insight into how generation quality changes
    as more atoms are masked, but produces N extra image sets per ligand
    (potentially 20+ files for a ligand with many interaction sites).
    Default: YES.
""")
    per_mask = _ask_yes_no_default(
        "Generate per-mask-count outputs in addition to aggregated?",
        default=True,
    )

    print("""
  Option 2 — Deduplication when pooling IA + random SMILES:
    When combining interaction-aware and random predictions into one pool,
    the same SMILES string might appear in both. Deduplication counts it
    once (reflecting true unique chemical diversity). Keeping both copies
    preserves the information that two strategies independently found the
    same molecule, but introduces duplicates in the grid and similarity
    scores.
    Default: YES (deduplicate).
""")
    deduplicate = _ask_yes_no_default(
        "Deduplicate pooled IA + random SMILES?",
        default=True,
    )
    print("""
  Option 3 — Data Source:
    Which dataset would you like to analyze?
    [1] Standard Stage 2
    [2] Stage 2.5 (Random Pick)
    [3] Stage 2.7 (Multi-seed)
""")
    
    while True:
        choice = input("  Enter 1, 2, or 3 [Default: 1]: ").strip()
        if choice in ['1', '']:
            stage_choice = '2'
            stage_name = "Stage 2 (Standard)"
            break
        elif choice == '2':
            stage_choice = '25'
            stage_name = "Stage 2.5 (Random Pick)"
            break
        elif choice == '3':
            stage_choice = '27'
            stage_name = "Stage 2.7 (Multi-seed)"
            break
        else:
            print("  ❌ Invalid choice. Please enter 1, 2, or 3.")

    pool_choice = _ask_pool_choice()

    print(f"\n  Settings confirmed:")
    print(f"    Per-mask-count outputs : {'yes' if per_mask else 'no'}")
    print(f"    Deduplicate pool       : {'yes' if deduplicate else 'no'}")
    print(f"    Data Source            : {stage_name}")
    print(f"    Pool choice            : {pool_choice}")
    print("─" * 60 + "\n")

    return per_mask, deduplicate, stage_choice, pool_choice




# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_stage1_smiles(mask_calc_dir: str) -> Dict[str, str]:
    """
    Load original ligand SMILES from Stage-1 .meta.json files.
    Returns {ligand_id: original_smiles}.
    """
    result: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(mask_calc_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            lig  = meta.get("ligand", {})
            key  = f"{lig.get('resname','?')}-{lig.get('chain','?')}-{lig.get('resseq','?')}"
            smi  = meta.get("smiles", "")
            if smi and Chem.MolFromSmiles(smi) is not None:
                result[key] = smi
        except Exception as e:
            print(f"  ⚠️  Could not read Stage-1 JSON {os.path.basename(path)}: {e}")
    return result


def _read_smiles_from_txt(path: str) -> List[str]:
    """
    Read valid SMILES (one per line) from a Stage-2 prediction txt file.
    Lines that RDKit cannot parse are silently skipped (Assumption A7).
    """
    valid = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                smi = line.strip()
                if smi and Chem.MolFromSmiles(smi) is not None:
                    valid.append(smi)
    except Exception as e:
        print(f"  ⚠️  Could not read {path}: {e}")
    return valid


def load_ligand_predictions(
    lig_dir: str,
) -> Dict[int, Dict[str, List[str]]]:
    """
    Discover ia_mask*.txt and rand_mask*.txt files under lig_dir.
    Returns {mask_count: {"ia": [smiles…], "rand": [smiles…]}}.
    """
    result: Dict[int, Dict[str, List[str]]] = {}

    ia_pattern   = os.path.join(lig_dir, "ia_mask*.txt")
    rand_pattern = os.path.join(lig_dir, "rand_mask*.txt")

    def _extract_count(path: str) -> Optional[int]:
        m = re.search(r"mask(\d+)\.txt$", os.path.basename(path))
        return int(m.group(1)) if m else None

    for path in sorted(glob.glob(ia_pattern)):
        mc = _extract_count(path)
        if mc is None:
            continue
        result.setdefault(mc, {"ia": [], "rand": []})
        result[mc]["ia"] = _read_smiles_from_txt(path)

    for path in sorted(glob.glob(rand_pattern)):
        mc = _extract_count(path)
        if mc is None:
            continue
        result.setdefault(mc, {"ia": [], "rand": []})
        result[mc]["rand"] = _read_smiles_from_txt(path)

    return result


def pool_smiles(
    ia_list: List[str],
    rand_list: List[str],
    deduplicate: bool,
) -> List[str]:
    """
    Combine IA and random SMILES into one list.
    If deduplicate=True, each unique SMILES string appears once (Assumption A5).
    """
    combined = ia_list + rand_list
    if deduplicate:
        seen, out = set(), []
        for smi in combined:
            if smi not in seen:
                seen.add(smi)
                out.append(smi)
        return out
    return combined


# ════════════════════════════════════════════════════════════════════════════
#  FINGERPRINTS & SIMILARITY
# ════════════════════════════════════════════════════════════════════════════

# Built once, not once per call. Constructing a MorganGenerator costs ~0.27 ms
# against ~0.06 ms for the fingerprint itself, so rebuilding it per molecule was
# ~6x the real work -- and Stage 9 calls this twice per scored molecule.
_MORGAN_GEN = None


def _get_morgan_generator():
    """The shared r=2 / 2048-bit Morgan generator, built on first use."""
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return _MORGAN_GEN


def _morgan_fp(smiles: str):
    """Return Morgan fingerprint (r=2, 2048 bits) or None if SMILES invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _get_morgan_generator().GetFingerprint(mol)


def pairwise_tanimoto(smiles_list: List[str]) -> List[float]:
    """
    Compute Tanimoto similarity for every unique pair in smiles_list.
    Returns a flat list of scores.
    """
    fps = [_morgan_fp(s) for s in smiles_list]
    fps = [fp for fp in fps if fp is not None]
    scores = []
    for fp_a, fp_b in combinations(fps, 2):
        scores.append(TanimotoSimilarity(fp_a, fp_b))
    return scores


def vs_original_tanimoto(smiles_list: List[str], original_smiles: str) -> List[float]:
    """
    Compute Tanimoto similarity of each molecule in smiles_list against
    the original ligand SMILES.
    """
    orig_fp = _morgan_fp(original_smiles)
    if orig_fp is None:
        return []
    scores = []
    for smi in smiles_list:
        fp = _morgan_fp(smi)
        if fp is not None:
            scores.append(TanimotoSimilarity(orig_fp, fp))
    return scores


# ════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ════════════════════════════════════════════════════════════════════════════

def _canon(smi: str) -> Optional[str]:
    """Return canonical SMILES or None if unparseable."""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


# ── Strategy border colour constants ─────────────────────────────────────────
_BORDER_IA_ONLY   = (31,  119, 180)   # matplotlib blue  — IA only
_BORDER_RAND_ONLY = (214,  39,  40)   # matplotlib red   — random only
_BORDER_BOTH      = (148, 103, 189)   # medium purple    — both strategies


def _draw_cell_border(draw, idx: int, mols_per_row: int,
                      cell_w: int, cell_h: int,
                      color: tuple, bw: int) -> None:
    """Draw a thick coloured rectangle around cell at position idx."""
    row = idx // mols_per_row
    col = idx %  mols_per_row
    x0  = col * cell_w
    y0  = row * cell_h
    x1  = x0 + cell_w - 1
    y1  = y0 + cell_h - 1
    for i in range(bw):
        draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i], outline=color)


def draw_molecule_grid(
    smiles_list: List[str],
    title: str,
    out_path: str,
    mols_per_row: int = 5,
    original_smiles: Optional[str] = None,
    sub_img_size: tuple = (300, 300),
    ia_smiles: Optional[List[str]] = None,
    rand_smiles: Optional[List[str]] = None,
) -> bool:
    """
    Draw smiles_list as a molecule grid with PIL borders marking:

    Strategy borders  (drawn first, 5 px — merged grid only):
      BLUE   (31,119,180)  — molecule generated by IA masking only
      RED   (214, 39, 40)  — molecule generated by random masking only
      PURPLE(148,103,189)  — molecule generated by BOTH strategies
      Legend suffix added: ◆IA / ◆rand / ◆both

    Original-molecule border (drawn second, 8 px — takes priority):
      GREEN  (0,180,0)     — original ligand reproduced by ChemBERTa (★orig)
      ORANGE (255,140,0)   — original ligand NOT generated, prepended at pos 0 (⊕orig)

    Original border is always drawn on top of any strategy border (Ambiguity 3A).
    Comparison is done on canonical SMILES.

    Parameters
    ----------
    smiles_list     : molecules to display
    original_smiles : original unmasked ligand SMILES (may be None)
    ia_smiles       : list of SMILES generated by IA strategy (before merging)
    rand_smiles     : list of SMILES generated by random strategy (before merging)
    sub_img_size    : (width, height) per cell in pixels — must match grid call
    """
    # ── Canonical sets for strategy lookup ───────────────────────────────────
    ia_canon_set   : set = set()
    rand_canon_set : set = set()
    if ia_smiles is not None:
        ia_canon_set   = {_canon(s) for s in ia_smiles if _canon(s) is not None}
    if rand_smiles is not None:
        rand_canon_set = {_canon(s) for s in rand_smiles if _canon(s) is not None}
    has_strategy = bool(ia_canon_set or rand_canon_set)

    # ── Build mol list ────────────────────────────────────────────────────────
    mols, legends, canon_list = [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            legends.append(smi[:28] + ("…" if len(smi) > 28 else ""))
            canon_list.append(Chem.MolToSmiles(mol))

    if not mols:
        print(f"    ⚠️  No drawable molecules for '{title}'. Grid skipped.")
        return False

    if len(mols) > config.MAX_GRID_MOLS:
        print(
            f"    ⚠️  Pool has {len(mols)} molecules; grid capped at {config.MAX_GRID_MOLS} "
            f"(Assumption A3). Similarity analysis uses full pool."
        )
        mols       = mols[:config.MAX_GRID_MOLS]
        legends    = legends[:config.MAX_GRID_MOLS]
        canon_list = canon_list[:config.MAX_GRID_MOLS]

    # ── Strategy legend suffixes (before original-marker may overwrite) ───────
    if has_strategy:
        for idx, csmi in enumerate(canon_list):
            in_ia   = csmi in ia_canon_set
            in_rand = csmi in rand_canon_set
            if in_ia and in_rand:
                suffix = " ◆both"
            elif in_ia:
                suffix = " ◆IA"
            elif in_rand:
                suffix = " ◆rand"
            else:
                suffix = ""
            if suffix:
                legends[idx] = legends[idx].rstrip("…")[:22] + suffix

    # ── Determine original-molecule cell index and colour ─────────────────────
    orig_cell_idx: Optional[int] = None
    orig_border:   Optional[tuple] = None

    if original_smiles is not None:
        orig_canon = _canon(original_smiles)
        if orig_canon is not None:
            if orig_canon in canon_list:
                # Already reproduced — mark its existing cell in GREEN
                orig_cell_idx = canon_list.index(orig_canon)
                orig_border   = (0, 180, 0)
                legends[orig_cell_idx] = (
                    legends[orig_cell_idx].rstrip("…")[:22] + " ★orig"
                )
            else:
                # Not generated — prepend at position 0, mark in ORANGE
                orig_mol = Chem.MolFromSmiles(orig_canon)
                orig_lbl = orig_canon[:22] + " ⊕orig"
                mols.insert(0, orig_mol)
                legends.insert(0, orig_lbl)
                canon_list.insert(0, orig_canon)
                orig_cell_idx = 0
                orig_border   = (255, 140, 0)
                if len(mols) > config.MAX_GRID_MOLS + 1:
                    mols       = mols[:config.MAX_GRID_MOLS + 1]
                    legends    = legends[:config.MAX_GRID_MOLS + 1]
                    canon_list = canon_list[:config.MAX_GRID_MOLS + 1]

    # ── Draw grid ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # NOTE (Colab/local): MolsToGridImage return type varies by RDKit version.
    result = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
    )
    if isinstance(result, bytes):
        png_bytes = result
    elif hasattr(result, "save"):          # PIL Image
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        png_bytes = buf.getvalue()
    elif hasattr(result, "data"):           # IPython.display.Image (Colab)
        png_bytes = result.data
    else:
        raise TypeError(
            f"Unexpected image type returned by RDKit: {type(result)}"
        )

    # ── Draw PIL borders ──────────────────────────────────────────────────────
    needs_border = (has_strategy and (ia_canon_set or rand_canon_set)) or orig_cell_idx is not None
    if needs_border:
        try:
            from PIL import Image as PilImage, ImageDraw as PilDraw
            img      = PilImage.open(io.BytesIO(png_bytes)).convert("RGB")
            draw_obj = PilDraw.Draw(img)
            cell_w, cell_h = sub_img_size

            # Pass 1 — strategy borders (5 px, drawn first)
            if has_strategy:
                for idx, csmi in enumerate(canon_list):
                    if idx == orig_cell_idx:
                        continue   # will be overdrawn by orig border in pass 2
                    in_ia   = csmi in ia_canon_set
                    in_rand = csmi in rand_canon_set
                    if in_ia and in_rand:
                        color = _BORDER_BOTH
                    elif in_ia:
                        color = _BORDER_IA_ONLY
                    elif in_rand:
                        color = _BORDER_RAND_ONLY
                    else:
                        continue
                    _draw_cell_border(draw_obj, idx, mols_per_row,
                                      cell_w, cell_h, color, bw=5)

            # Pass 2 — original-molecule border (8 px, drawn on top — takes priority)
            if orig_cell_idx is not None and orig_border is not None:
                _draw_cell_border(draw_obj, orig_cell_idx, mols_per_row,
                                  cell_w, cell_h, orig_border, bw=8)

            buf2 = io.BytesIO()
            img.save(buf2, format="PNG")
            png_bytes = buf2.getvalue()
        except Exception as e:
            print(f"    ⚠️  Could not draw borders: {e}")

    with open(out_path, "wb") as f:
        f.write(png_bytes)
    return True


# ════════════════════════════════════════════════════════════════════════════
#  INCREMENTAL GENERATION CURVE
# ════════════════════════════════════════════════════════════════════════════

def plot_mask_count_curve(
    mask_counts:  List[int],
    ia_counts:    List[int],
    rand_counts:  List[int],
    lig_id:       str,
    out_path:     str,
) -> bool:
    """
    Plot number of valid unique molecules (Y) vs number of masked atoms (X).

    Two lines:
      Blue  — Interaction-Aware (IA) masking
      Red   — Random masking

    Y = len(set(predictions[mc]["ia"])) per mask_count step (Option A:
    unique within each step independently, not cumulative).

    Saved as PNG to out_path. Returns True on success.
    """
    if not mask_counts:
        print(f"    ⚠️  {lig_id}: no mask-count data for curve plot. Skipped.")
        return False

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#FAFAFA")

    ax.plot(
        mask_counts, ia_counts,
        color="#1F77B4", marker="o", linewidth=2, markersize=6,
        label="Interaction-Aware (IA)",
    )
    ax.plot(
        mask_counts, rand_counts,
        color="#D62728", marker="s", linewidth=2, markersize=6,
        linestyle="--", label="Random masking",
    )

    ax.set_xlabel("Number of Masked Atoms (mask_count)", fontsize=11)
    ax.set_ylabel("Unique Valid Molecules Generated", fontsize=11)
    ax.set_title(
        f"Incremental Generation Curve — {lig_id}\nUnique valid molecules per masking step",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(mask_counts)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_facecolor("#FAFAFA")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"    🖼  {os.path.basename(out_path)}")
    return True


# ════════════════════════════════════════════════════════════════════════════
#  HISTOGRAMS
# ════════════════════════════════════════════════════════════════════════════

def _save_histogram(
    scores: List[float],
    xlabel: str,
    title: str,
    out_path: str,
    color: str,
    n_total_mols: int,
) -> bool:
    """
    Plot and save a histogram of Tanimoto scores.
    Returns True on success.
    """
    if not scores:
        print(f"    ⚠️  No similarity scores for '{title}'. Histogram skipped.")
        return False

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=min(20, max(5, len(scores) // 3)),
            color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title(
        f"{title}\n(n molecules = {n_total_mols}, n scores = {len(scores)})",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    mean_val = float(np.mean(scores))
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
               label=f"mean = {mean_val:.3f}")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_pairwise_histogram(
    smiles_list: List[str],
    title: str,
    out_path: str,
) -> bool:
    """
    Plot pairwise within-set Tanimoto distribution.
    Skipped (returns False) if fewer than 2 molecules (Assumption A2).
    """
    if len(smiles_list) < 2:
        print(
            f"    ⚠️  '{title}': only {len(smiles_list)} molecule(s) — "
            "need ≥ 2 for pairwise histogram. Skipped."
        )
        return False
    scores = pairwise_tanimoto(smiles_list)
    return _save_histogram(
        scores=scores,
        xlabel="Pairwise Tanimoto Similarity",
        title=title,
        out_path=out_path,
        color="#1f77b4",
        n_total_mols=len(smiles_list),
    )


def plot_vs_original_histogram(
    smiles_list: List[str],
    original_smiles: Optional[str],
    title: str,
    out_path: str,
) -> bool:
    """
    Plot each-molecule-vs-original Tanimoto distribution.
    Skipped if original_smiles is None (Assumption A4).
    """
    if not original_smiles:
        print(f"    ⚠️  '{title}': no original SMILES available. vs-original histogram skipped.")
        return False
    if not smiles_list:
        print(f"    ⚠️  '{title}': empty pool. vs-original histogram skipped.")
        return False
    scores = vs_original_tanimoto(smiles_list, original_smiles)
    return _save_histogram(
        scores=scores,
        xlabel="Tanimoto Similarity to Original Ligand",
        title=title,
        out_path=out_path,
        color="#ff7f0e",
        n_total_mols=len(smiles_list),
    )


# ════════════════════════════════════════════════════════════════════════════
#  PER-GROUP ANALYSIS  (one grid + two histograms)
# ════════════════════════════════════════════════════════════════════════════

def _dedup(smiles_list: List[str]) -> List[str]:
    """Return a list with duplicate SMILES strings removed, preserving order."""
    seen: set = set()
    out: List[str] = []
    for s in smiles_list:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def run_analysis_for_group(
    smiles_pool: List[str],
    original_smiles: Optional[str],
    label: str,
    out_dir: str,
    ia_list: Optional[List[str]] = None,
    rand_list: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    Produce grid + pairwise histogram + vs-original histogram for one group.

    Always produces (existing outputs — unchanged):
      molecule_grid.png              — all molecules in the merged pool
      histogram_pairwise.png         — pairwise Tanimoto within merged pool
      histogram_vs_original.png      — merged pool vs original ligand

    Additionally produces per-strategy histograms when ia_list / rand_list
    are supplied (new outputs):
      histogram_pairwise_ia.png      — pairwise within IA-only molecules
      histogram_pairwise_rand.png    — pairwise within random-only molecules
      histogram_vs_original_ia.png   — IA-only molecules vs original ligand
      histogram_vs_original_rand.png — random-only molecules vs original ligand

    Parameters
    ----------
    smiles_pool     : merged pool (IA + random, possibly deduplicated)
    original_smiles : original unmasked ligand SMILES (may be None)
    label           : human-readable label used in plot titles
    out_dir         : directory where files are saved
    ia_list         : IA-only SMILES (before merging); None to skip per-strategy plots
    rand_list       : random-only SMILES (before merging); None to skip per-strategy plots

    Returns
    -------
    dict with boolean values for each output file key.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(smiles_pool)
    print(f"  Group: {label}  |  pool size: {n}")

    # ── Existing outputs (merged pool) ────────────────────────────────────────
    # ia_list / rand_list may be None when called without strategy data;
    # draw_molecule_grid handles None gracefully (no strategy borders drawn).
    grid_ok = draw_molecule_grid(
        smiles_list     = smiles_pool,
        title           = label,
        out_path        = os.path.join(out_dir, "molecule_grid.png"),
        original_smiles = original_smiles,
        ia_smiles       = ia_list,
        rand_smiles     = rand_list,
    )

    # Compute scores once — used for histogram AND returned for summary boxplot
    pw_scores = pairwise_tanimoto(smiles_pool) if len(smiles_pool) >= 2 else []
    vo_scores = (vs_original_tanimoto(smiles_pool, original_smiles)
                 if original_smiles and smiles_pool else [])

    pw_ok = plot_pairwise_histogram(
        smiles_list = smiles_pool,
        title       = f"Pairwise Tanimoto — {label}",
        out_path    = os.path.join(out_dir, "histogram_pairwise.png"),
    )

    vo_ok = plot_vs_original_histogram(
        smiles_list     = smiles_pool,
        original_smiles = original_smiles,
        title           = f"vs Original Ligand — {label}",
        out_path        = os.path.join(out_dir, "histogram_vs_original.png"),
    )

    result = {
        "grid": grid_ok, "pairwise": pw_ok, "vs_original": vo_ok,
        # Raw scores for summary boxplot (Option A)
        "pairwise_scores":    pw_scores,
        "vs_original_scores": vo_scores,
    }

    # ── New per-strategy outputs ──────────────────────────────────────────────
    # Deduplicate each strategy list independently (the same SMILES can appear
    # across multiple mask_count files when aggregating all mask counts).
    if ia_list is not None:
        ia_dedup = _dedup(ia_list)
        result["grid_ia"] = draw_molecule_grid(
            smiles_list     = ia_dedup,
            title           = f"{label} (IA only)",
            out_path        = os.path.join(out_dir, "molecule_grid_ia.png"),
            original_smiles = original_smiles,
        )
        ia_pw_scores = pairwise_tanimoto(ia_dedup) if len(ia_dedup) >= 2 else []
        ia_vo_scores = (vs_original_tanimoto(ia_dedup, original_smiles)
                        if original_smiles and ia_dedup else [])
        result["pairwise_ia"] = plot_pairwise_histogram(
            smiles_list = ia_dedup,
            title       = f"Pairwise Tanimoto (IA only) — {label}",
            out_path    = os.path.join(out_dir, "histogram_pairwise_ia.png"),
        )
        result["vs_original_ia"] = plot_vs_original_histogram(
            smiles_list     = ia_dedup,
            original_smiles = original_smiles,
            title           = f"vs Original Ligand (IA only) — {label}",
            out_path        = os.path.join(out_dir, "histogram_vs_original_ia.png"),
        )
        result["pairwise_ia_scores"]    = ia_pw_scores
        result["vs_original_ia_scores"] = ia_vo_scores

    if rand_list is not None:
        rand_dedup = _dedup(rand_list)
        result["grid_rand"] = draw_molecule_grid(
            smiles_list     = rand_dedup,
            title           = f"{label} (random only)",
            out_path        = os.path.join(out_dir, "molecule_grid_rand.png"),
            original_smiles = original_smiles,
        )
        rand_pw_scores = pairwise_tanimoto(rand_dedup) if len(rand_dedup) >= 2 else []
        rand_vo_scores = (vs_original_tanimoto(rand_dedup, original_smiles)
                         if original_smiles and rand_dedup else [])
        result["pairwise_rand"] = plot_pairwise_histogram(
            smiles_list = rand_dedup,
            title       = f"Pairwise Tanimoto (random only) — {label}",
            out_path    = os.path.join(out_dir, "histogram_pairwise_rand.png"),
        )
        result["vs_original_rand"] = plot_vs_original_histogram(
            smiles_list     = rand_dedup,
            original_smiles = original_smiles,
            title           = f"vs Original Ligand (random only) — {label}",
            out_path        = os.path.join(out_dir, "histogram_vs_original_rand.png"),
        )
        result["pairwise_rand_scores"]    = rand_pw_scores
        result["vs_original_rand_scores"] = rand_vo_scores

    return result


# ════════════════════════════════════════════════════════════════════════════
#  SAFE FILENAME HELPER
# ════════════════════════════════════════════════════════════════════════════

def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY BOXPLOTS (all ligands side by side)
# ════════════════════════════════════════════════════════════════════════════

def plot_summary_boxplot_stage3(
    all_results: Dict[str, dict],
    strategy: str,
    stage3_dir: str,
) -> None:
    """
    Produce two summary boxplots for a given strategy, placing all ligands
    side by side so cross-ligand diversity is immediately visible.

    strategy : "ia", "rand", or "both" (merged pool)

    Outputs (saved to stage3_dir/summary_plot_{strategy}/):
      boxplot_pairwise_tanimoto_{strategy}.png
          X = ligand ID, Y = pairwise Tanimoto scores
      boxplot_vs_original_tanimoto_{strategy}.png
          X = ligand ID, Y = vs-original Tanimoto scores
    """
    # Map strategy to the correct score keys in the result dict
    if strategy == "ia":
        pw_key = "pairwise_ia_scores"
        vo_key = "vs_original_ia_scores"
        tag    = "IA only"
    elif strategy == "rand":
        pw_key = "pairwise_rand_scores"
        vo_key = "vs_original_rand_scores"
        tag    = "Random only"
    else:
        pw_key = "pairwise_scores"
        vo_key = "vs_original_scores"
        tag    = "IA + Random (merged)"

    out_dir = os.path.join(stage3_dir, f"summary_plot_{strategy}")
    os.makedirs(out_dir, exist_ok=True)

    lig_ids = sorted(all_results.keys())

    def _extract(key: str) -> Tuple[List[str], List[List[float]]]:
        labels, data = [], []
        for lid in lig_ids:
            agg = all_results[lid].get("aggregated", {})
            scores = agg.get(key, [])
            if scores:
                labels.append(_safe(lid))
                data.append(scores)
        return labels, data

    for metric, key, xlabel, color, fname in [
        ("Pairwise Tanimoto",       pw_key,
         "Pairwise Tanimoto Similarity",
         "#1f77b4",
         f"boxplot_pairwise_tanimoto_{strategy}.png"),
        ("vs-Original Tanimoto",    vo_key,
         "Tanimoto Similarity to Original Ligand",
         "#ff7f0e",
         f"boxplot_vs_original_tanimoto_{strategy}.png"),
    ]:
        labels, data = _extract(key)
        if not data:
            print(f"  ⚠️  Summary boxplot ({metric}, {strategy}): no data. Skipped.")
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 6),
                               facecolor="#FAFAFA")
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels,
                        medianprops=dict(color="red", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xlabel("Ligand Group", fontsize=11)
        ax.set_ylabel(xlabel, fontsize=11)
        ax.set_title(
            f"{metric} — All Ligands  ({tag})\n"
            f"Stage 3 summary  (N={len(labels)} groups)",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylim(0, 1)
        ax.axhline(0.4, color="#AAAAAA", linestyle="--",
                   linewidth=0.8, label="T = 0.40 reference")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.set_facecolor("#FAFAFA")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  🖼  {os.path.basename(out_path)}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_stage3(
    pred_dir:       Optional[str] = None,
    mask_calc_dir:  Optional[str] = None,
    stage3_dir:     Optional[str] = None,
    per_mask_count: bool = True,
    deduplicate:    bool = True,
    pool_choice:    str  = "both",
) -> Dict[str, dict]:
    """
    Run Stage-3 analysis.

    Parameters
    ----------
    pred_dir       : Stage-2 predictions folder  (default: config.PRED_DIR)
    mask_calc_dir  : Stage-1 JSON folder          (default: config.MASK_CALC_OUTDIR)
    stage3_dir     : output root for this stage   (default: config.STAGE3_DIR)
    per_mask_count : also produce per-step outputs
    deduplicate    : deduplicate pooled IA + random SMILES
    pool_choice    : "ia", "rand", or "both" — which strategies to include

    Returns
    -------
    Dict keyed by ligand_id → analysis summary dict
    """
    pred_dir      = pred_dir      or config.PRED_DIR
    mask_calc_dir = mask_calc_dir or config.MASK_CALC_OUTDIR
    stage3_dir    = stage3_dir    or config.STAGE3_DIR

    os.makedirs(stage3_dir, exist_ok=True)

    # ── Load original SMILES from Stage-1 JSONs ───────────────────────────────
    original_smiles_map = load_stage1_smiles(mask_calc_dir)
    print(f"  Stage-1 original SMILES loaded: {len(original_smiles_map)} ligand(s)")

    # ── Discover ligand sub-folders in PRED_DIR ───────────────────────────────
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
        lig_id = os.path.basename(lig_dir)
        orig   = original_smiles_map.get(lig_id)
        if orig is None:
            print(f"\n  ⚠️  {lig_id}: no Stage-1 SMILES found — vs-original histograms will be skipped.")

        print(f"\n{'='*60}")
        print(f"Ligand: {lig_id}")
        print(f"{'='*60}")

        # Load all predictions for this ligand
        predictions = load_ligand_predictions(lig_dir)
        if not predictions:
            print(f"  ⚠️  No prediction txt files found. Skipping.")
            continue

        mask_counts = sorted(predictions.keys())
        lig_out_dir = os.path.join(stage3_dir, _safe(lig_id))

        # ── Aggregated: pool all mask counts, filtered by pool_choice ─────────
        all_ia   = ([smi for mc in mask_counts for smi in predictions[mc]["ia"]]
                    if pool_choice in ("ia", "both") else [])
        all_rand = ([smi for mc in mask_counts for smi in predictions[mc]["rand"]]
                    if pool_choice in ("rand", "both") else [])
        agg_pool = pool_smiles(all_ia, all_rand, deduplicate)

        agg_result = run_analysis_for_group(
            smiles_pool     = agg_pool,
            original_smiles = orig,
            label           = f"{lig_id} — aggregated (all masks, N={len(mask_counts)}, pool={pool_choice})",
            out_dir         = os.path.join(lig_out_dir, "aggregated"),
            ia_list         = all_ia if pool_choice in ("ia", "both") else None,
            rand_list       = all_rand if pool_choice in ("rand", "both") else None,
        )

        per_mc_results: Dict[int, dict] = {}

        # ── Per-mask-count ────────────────────────────────────────────────────
        if per_mask_count:
            for mc in mask_counts:
                mc_ia   = predictions[mc]["ia"]   if pool_choice in ("ia",   "both") else []
                mc_rand = predictions[mc]["rand"]  if pool_choice in ("rand", "both") else []
                pool = pool_smiles(mc_ia, mc_rand, deduplicate)
                mc_result = run_analysis_for_group(
                    smiles_pool     = pool,
                    original_smiles = orig,
                    label           = f"{lig_id} — mask_count={mc} ({pool_choice})",
                    out_dir         = os.path.join(lig_out_dir, f"mask_{mc:03d}"),
                    ia_list         = mc_ia   if pool_choice in ("ia",   "both") else None,
                    rand_list       = mc_rand if pool_choice in ("rand", "both") else None,
                )
                per_mc_results[mc] = mc_result

        # ── Incremental generation curve ──────────────────────────────────────
        # Compute unique valid count per step independently (Option A).
        ia_counts   = [len(set(predictions[mc]["ia"]))   for mc in mask_counts]
        rand_counts = [len(set(predictions[mc]["rand"])) for mc in mask_counts]
        curve_path  = os.path.join(lig_out_dir, "incremental_generation_curve.png")
        plot_mask_count_curve(
            mask_counts = mask_counts,
            ia_counts   = ia_counts,
            rand_counts = rand_counts,
            lig_id      = lig_id,
            out_path    = curve_path,
        )

        all_results[lig_id] = {
            "aggregated":    agg_result,
            "per_mask_count": per_mc_results,
            "n_mask_counts": len(mask_counts),
            "agg_pool_size": len(agg_pool),
            "curve_path":    curve_path,
        }

    # ── Summary boxplots: all ligands side by side ───────────────────────────
    # Dispatch is driven directly by pool_choice (not inferred from score keys):
    #   pool_choice = "ia"   → 2 plots in summary_plot_ia/
    #   pool_choice = "rand" → 2 plots in summary_plot_rand/
    #   pool_choice = "both" → 4 plots: summary_plot_ia/ + summary_plot_rand/
    if pool_choice in ("ia", "both"):
        plot_summary_boxplot_stage3(all_results, "ia", stage3_dir)
    if pool_choice in ("rand", "both"):
        plot_summary_boxplot_stage3(all_results, "rand", stage3_dir)

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED TEST
# ════════════════════════════════════════════════════════════════════════════

# Realistic SMILES pools to test with (diverse drug-like molecules)
_TEST_IA_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "OC(=O)c1ccccc1O",                # salicylic acid
]
_TEST_RAND_SMILES = [
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin (duplicate — tests deduplication)
    "c1ccc2ccccc2c1",                  # naphthalene
]
_TEST_ORIGINAL_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin as "original"


def _run_test() -> bool:
    import shutil

    print("\n" + "=" * 60)
    print("STAGE 3 SELF-TEST")
    print("=" * 60)

    # ── Wipe and recreate config.STAGE3_DIR/test/ (Option A) ─────────────────
    td = os.path.join(config.STAGE3_DIR, "test")
    if os.path.exists(td):
        shutil.rmtree(td)
        print(f"  🗑  Wiped existing test dir: {td}")
    os.makedirs(td)
    print(f"  📁 Created fresh test dir:  {td}\n")

    pred_dir      = os.path.join(td, "preds")
    mask_calc_dir = os.path.join(td, "stage1")
    stage3_dir    = os.path.join(td, "stage3_a")   # per_mask_count=True run
    stage3_dir_b  = os.path.join(td, "stage3_b")   # per_mask_count=False run

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

    # ── Write synthetic Stage-2 txt files ────────────────────────────────────
    lig_id  = "ASP-A-1"
    lig_dir = os.path.join(pred_dir, _safe(lig_id))
    os.makedirs(lig_dir)

    for mc in range(1, 4):        # 3 mask steps
        with open(os.path.join(lig_dir, f"ia_mask{mc:03d}.txt"), "w") as f:
            f.write("\n".join(_TEST_IA_SMILES) + "\n")
        with open(os.path.join(lig_dir, f"rand_mask{mc:03d}.txt"), "w") as f:
            f.write("\n".join(_TEST_RAND_SMILES) + "\n")

    # ── Write synthetic Stage-1 JSON ──────────────────────────────────────────
    os.makedirs(mask_calc_dir)
    s1_meta = {
        "smiles": _TEST_ORIGINAL_SMILES,
        "ligand": {"resname": "ASP", "chain": "A", "resseq": 1},
        "masked_atom_indices": [0, 3],
    }
    with open(os.path.join(mask_calc_dir, "ASP-A-1.meta.json"), "w") as f:
        json.dump(s1_meta, f)

    # ── Run A: per_mask_count=True, deduplicate=True ──────────────────────────
    results = run_stage3(
        pred_dir       = pred_dir,
        mask_calc_dir  = mask_calc_dir,
        stage3_dir     = stage3_dir,
        per_mask_count = True,
        deduplicate    = True,
    )

    check(lig_id in results, f"'{lig_id}' present in results")

    if lig_id in results:
        r   = results[lig_id]
        agg = r["aggregated"]

        check(r["n_mask_counts"] == 3,
              "n_mask_counts == 3")
        check(agg["grid"],
              "aggregated grid produced")
        check(agg["pairwise"],
              "aggregated pairwise histogram produced")
        check(agg["vs_original"],
              "aggregated vs-original histogram produced")

        # Check files exist on disk
        agg_dir = os.path.join(stage3_dir, _safe(lig_id), "aggregated")
        check(os.path.exists(os.path.join(agg_dir, "molecule_grid.png")),
              "molecule_grid.png on disk")
        check(os.path.exists(os.path.join(agg_dir, "histogram_pairwise.png")),
              "histogram_pairwise.png on disk")
        check(os.path.exists(os.path.join(agg_dir, "histogram_vs_original.png")),
              "histogram_vs_original.png on disk")

        # Check per-mask-count outputs
        for mc in range(1, 4):
            mc_dir = os.path.join(stage3_dir, _safe(lig_id), f"mask_{mc:03d}")
            check(os.path.exists(os.path.join(mc_dir, "molecule_grid.png")),
                  f"mask_{mc:03d}/molecule_grid.png on disk")

        # Deduplication: aspirin appears in both IA and RAND lists →
        # deduplicated pool should be smaller than raw concatenation
        agg_pool_size = r["agg_pool_size"]
        raw_size      = 3 * (len(_TEST_IA_SMILES) + len(_TEST_RAND_SMILES))
        check(agg_pool_size < raw_size,
              f"deduplication reduced pool: {agg_pool_size} < {raw_size} (raw)")

    # ── Run B: per_mask_count=False ───────────────────────────────────────────
    run_stage3(
        pred_dir       = pred_dir,
        mask_calc_dir  = mask_calc_dir,
        stage3_dir     = stage3_dir_b,
        per_mask_count = False,
        deduplicate    = True,
    )
    mc_dir_b = os.path.join(stage3_dir_b, _safe(lig_id), "mask_001")
    check(not os.path.exists(mc_dir_b),
          "per-mask-count dir absent when per_mask_count=False")

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
    print("STAGE 3: CHEMICAL SIMILARITY ANALYSIS")
    print("=" * 60)

    print("""
  Before running the full analysis (which reads all Stage-2 prediction
  files and Stage-1 JSONs), you can run a smoke test instead.

  The smoke test:
    • Does NOT need any Stage-1 or Stage-2 output files.
    • Creates synthetic predictions for one ligand (aspirin) with
      hard-coded SMILES (aspirin, ibuprofen, paracetamol, caffeine).
    • Runs the full grid + histogram analysis with per_mask_count=True
      and deduplication=True, then re-runs with per_mask_count=False.
    • Saves all outputs to:
        {test_dir}
      (this directory is wiped clean at the start of every test run).
    • Prints PASS / FAIL for each assertion.
""".format(test_dir=os.path.join(config.STAGE3_DIR, "test")))

    run_test = _ask_yes_no_default("Run the smoke test?", default=False)

    if run_test:
        ok = _run_test()
        sys.exit(0 if ok else 1)

    per_mask_count, deduplicate, stage_choice, pool_choice = ask_runtime_options()
    # 2. Route paths based on user choice pulling directly from config
    if stage_choice == '25':
        in_dir  = config.STAGE25_PRED_DIR  # <-- Change if your config name differs
        out_dir = config.STAGE3_DIR_2_5
    elif stage_choice == '27':
        in_dir  = config.STAGE27_PRED_DIR  # <-- Change if your config name differs
        out_dir = config.STAGE3_DIR_2_7
    else:
        in_dir  = config.PRED_DIR
        out_dir = config.STAGE3_DIR

    # 3. Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    # 4. Run the stage with dynamic paths
    results = run_stage3(
        pred_dir       = in_dir,
        stage3_dir     = out_dir,
        per_mask_count = per_mask_count,
        deduplicate    = deduplicate,
        pool_choice    = pool_choice,
    )

    print("\n" + "=" * 60)
    print("✅ Stage 3 complete.")
    print(f"   Outputs saved to: {out_dir}")
    print(f"   Ligands analysed: {len(results)}")
    for lig_id, r in sorted(results.items()):
        agg = r["aggregated"]
        print(
            f"   • {lig_id}  pool={r['agg_pool_size']}  "
            f"grid={'✓' if agg['grid'] else '✗'}  "
            f"pairwise={'✓' if agg['pairwise'] else '✗'}  "
            f"vs_orig={'✓' if agg['vs_original'] else '✗'}  "
            f"mask_steps={r['n_mask_counts']}"
        )


if __name__ == "__main__":
    main()

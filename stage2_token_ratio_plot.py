# -*- coding: utf-8 -*-
"""
stage2_token_ratio_plot.py
==========================
Shared Plot 2 for Stages 2.5 and 2.7 (and optionally Stage 2):

    x-axis : % of full-SMILES BPE tokens masked (per strategy)
    y-axis : (unique valid SMILES) / total_samples

Import this module directly from whichever stage script you run — it does not
pull in ChemBERTa or generation logic.
"""

from typing import List, Optional

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage2_molecule_generation import safe_filename


def plot_token_ratio_results(
    lig_id: str,
    ia_token_pcts: List[float],
    rand_token_pcts: List[float],
    ia_valids: List[int],
    rand_valids: List[int],
    plot_dir: str,
    total_samples: int,
    denom_label: Optional[str] = None,
) -> str:
    """
    Save a PNG showing unique-valid yield vs token-masking percentage.

    x-axis : % of full-SMILES BPE tokens masked (per strategy).
    y-axis : (unique valid SMILES) / total_samples, where total_samples is the
             number of candidate draws that fed the plotted count (e.g.
             num_samples for Stage 2.5, num_samples × n_seeds for Stage 2.7).

    Two lines (interaction-aware vs random). Points sorted by x.
    Returns the saved PNG path.
    """
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{safe_filename(lig_id)}_token_ratio.png")

    denom      = max(int(total_samples), 1)
    ia_ratio   = [v / denom for v in ia_valids]
    rand_ratio = [v / denom for v in rand_valids]

    def _sorted_xy(xs: List[float], ys: List[float]):
        pairs = sorted(zip(xs, ys), key=lambda p: p[0])
        return [p[0] for p in pairs], [p[1] for p in pairs]

    ia_x,   ia_y   = _sorted_xy(ia_token_pcts,   ia_ratio)
    rand_x, rand_y = _sorted_xy(rand_token_pcts, rand_ratio)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        ia_x, ia_y,
        marker="o", linewidth=2, color="#1f77b4",
        label="Interaction-Aware (Stage 1)",
    )
    ax.plot(
        rand_x, rand_y,
        marker="s", linewidth=2, color="#ff7f0e", linestyle="--",
        label="Random (Stage 1.5)",
    )

    ax.set_xlabel("Tokens masked (% of full-SMILES BPE tokens)", fontsize=13)
    ax.set_ylabel("Unique valid / total samples", fontsize=13)
    title_denom = denom_label or f"denominator = {denom}"
    ax.set_title(
        f"Token-masking yield: {lig_id}\n({title_denom})",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼  Plot saved: {out_path}")
    return out_path

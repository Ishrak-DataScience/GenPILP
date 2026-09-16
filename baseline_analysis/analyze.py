# -*- coding: utf-8 -*-
"""
analyze.py
==========
Step 5: the four questions the baseline exists to answer.

  1. Are the predictions better than the redocked parent?
  2. Are they chemically diverse?
  3. Are they toxic?
  4. Are they drug-like?
  ... and, across all four: does the MASKING STRATEGY change the answer?

Reads predictions.csv + docking_summary.csv, writes:

    per_molecule.csv       one row per prediction: properties + best docking pose
    per_complex_delta.csv  per (complex, arm): median score and delta vs the parent
    arm_summary.csv        per arm: every headline statistic with its n
    summary_stats.csv      the same numbers in long form (metric, arm, value)
    figures/*.png          the five figures described below
    RESULTS.md             the written summary, tables included

HOW SCORES ARE JOINED
----------------------
run_docking.py docks each DISTINCT molecule once per complex, so a prediction is
matched to its docking row by (complex_id, canonical SMILES), not by (arm, k) --
two arms that propose the same molecule share one docking result, which is the
honest treatment (same molecule, same pocket, same score) and is why the
per-molecule table can show one docking row backing several prediction rows.

WHAT "BETTER" MEANS
--------------------
Primary metric is GNINA CNNaffinity (higher = better), matching Stage 6's own
ranking (A12). minimizedAffinity (Vinardo, kcal/mol, lower = better) is carried
alongside and reported as an improvement (parent - prediction) so that for BOTH
metrics a positive number means the prediction won. Per molecule, the best pose
is used (max CNNaffinity), not pose 1.

The reference is arm="parent_redock": the crystal ligand's own SMILES, embedded
and docked exactly like a prediction. Complexes whose redock RMSD exceeds
--rmsd-cutoff are FLAGGED and every headline statistic is reported twice, all
complexes and trustworthy-only: if the docking setup could not reproduce the
crystal pose, its deltas are not evidence about the model.

STATISTICS, AND WHY THESE ONES
-------------------------------
Docking deltas are neither normal nor independent (five predictions share a
complex, complexes differ wildly in baseline affinity), so:

  * per-complex aggregation first -- each complex contributes ONE median delta
    per arm, so a complex with many valid predictions cannot dominate;
  * Wilcoxon signed-rank on those per-complex deltas against 0 (is this arm
    better than its own reference?), which is paired and distribution-free;
  * Friedman test across the three arms on complexes where all three have a
    value (does the masking strategy matter at all?), then pairwise Wilcoxon
    with Holm-Bonferroni correction;
  * bootstrap 95% CI on the median (10k resamples, seeded) rather than a
    standard error that assumes symmetry.

Every test reports its n. With ~24 complexes these are small-sample tests and
the report says so rather than implying more power than exists.

HOW TO RUN
-----------
    python baseline_analysis/analyze.py --out-dir "$BASELINE_DIR"
    python baseline_analysis/analyze.py --no-docking     # properties only
    python baseline_analysis/analyze.py --test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_common as bc  # noqa: E402

# ── Palette ──────────────────────────────────────────────────────────────────
# The dataviz reference palette's categorical slots, in fixed slot order (never
# cycled, never recoloured when an arm drops out). The parent reference is NOT a
# series: it is muted ink, dashed, and always also labelled, so the reference is
# never identified by colour alone.
COLOR = {
    "plip_pos": "#2a78d6",   # slot 1, blue
    "plip_neg": "#eb6834",   # slot 2, orange
    "random":   "#1baf7a",   # slot 3, aqua
    "parent":   "#898781",   # muted ink
}
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"

PRIMARY = "CNNaffinity"      # higher = better (Stage 6 A12)
SECONDARY = "minimizedAffinity"   # kcal/mol, lower = better


def _f(x) -> Optional[float]:
    """Permissive float: '' / None / 'nan' -> None, so a missing value is never 0."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except ValueError:
        return None


def canonical(smiles: str) -> str:
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(m) if m is not None else ""


# ════════════════════════════════════════════════════════════════════════════
#  STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def median(xs: Sequence[float]) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def iqr(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 4:
        return (None, None)
    return (xs[len(xs) // 4], xs[(3 * len(xs)) // 4])


def bootstrap_median_ci(xs: Sequence[float], n_boot: int = 10000,
                        seed: int = 42) -> Tuple[Optional[float], Optional[float]]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 3:
        return (None, None)
    rng = random.Random(seed)
    meds = []
    for _ in range(n_boot):
        meds.append(median([xs[rng.randrange(len(xs))] for _ in range(len(xs))]))
    meds.sort()
    return (round(meds[int(0.025 * n_boot)], 3), round(meds[int(0.975 * n_boot)], 3))


def wilcoxon_vs_zero(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float], int]:
    """(statistic, p, n_nonzero). None when scipy is missing or n is too small."""
    xs = [x for x in xs if x is not None and x != 0]
    if len(xs) < 5:
        return (None, None, len(xs))
    try:
        from scipy.stats import wilcoxon
        st, p = wilcoxon(xs)
        return (float(st), float(p), len(xs))
    except Exception:
        return (None, None, len(xs))


def friedman(groups: Dict[str, Dict[str, float]]) -> Tuple[Optional[float], Optional[float], int]:
    """
    Friedman across arms on complexes where every arm has a value (the paired
    design). Returns (statistic, p, n_complexes).
    """
    arms = [a for a in bc.ARMS if a in groups]
    if len(arms) < 3:
        return (None, None, 0)
    common = set(groups[arms[0]])
    for a in arms[1:]:
        common &= set(groups[a])
    common = sorted(common)
    if len(common) < 5:
        return (None, None, len(common))
    try:
        from scipy.stats import friedmanchisquare
        cols = [[groups[a][c] for c in common] for a in arms]
        st, p = friedmanchisquare(*cols)
        return (float(st), float(p), len(common))
    except Exception:
        return (None, None, len(common))


def pairwise_wilcoxon(groups: Dict[str, Dict[str, float]]) -> List[dict]:
    """Paired arm-vs-arm Wilcoxon on shared complexes, Holm-Bonferroni corrected."""
    arms = [a for a in bc.ARMS if a in groups]
    out: List[dict] = []
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return out
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            common = sorted(set(groups[a]) & set(groups[b]))
            diffs = [groups[a][c] - groups[b][c] for c in common]
            nz = [d for d in diffs if d != 0]
            if len(nz) < 5:
                out.append({"arm_a": a, "arm_b": b, "n": len(common), "p": None,
                            "p_holm": None, "median_diff": median(diffs)})
                continue
            try:
                _, p = wilcoxon(nz)
            except Exception:
                p = None
            out.append({"arm_a": a, "arm_b": b, "n": len(common),
                        "p": None if p is None else float(p),
                        "p_holm": None, "median_diff": median(diffs)})
    tested = [o for o in out if o["p"] is not None]
    for rank, o in enumerate(sorted(tested, key=lambda d: d["p"])):
        o["p_holm"] = min(1.0, o["p"] * (len(tested) - rank))
    for rank in range(1, len(tested)):
        s = sorted(tested, key=lambda d: d["p"])
        s[rank]["p_holm"] = max(s[rank]["p_holm"], s[rank - 1]["p_holm"])   # monotone
    return out


# ── Chemistry ────────────────────────────────────────────────────────────────

def morgan(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(m)


def mean_pairwise_diversity(smiles_list: Sequence[str]) -> Tuple[Optional[float], int]:
    """
    Mean (1 - Tanimoto) over all distinct pairs: 0 = every molecule identical,
    1 = nothing in common. Returns (value, n_molecules_fingerprinted).
    """
    from rdkit.DataStructs import TanimotoSimilarity

    fps = [f for f in (morgan(s) for s in smiles_list) if f is not None]
    if len(fps) < 2:
        return (None, len(fps))
    tot = n = 0.0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            tot += 1.0 - TanimotoSimilarity(fps[i], fps[j])
            n += 1
    return (round(tot / n, 4), len(fps))


def murcko(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return ""
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════════════════
#  TABLE BUILDING
# ════════════════════════════════════════════════════════════════════════════

def best_pose_index(dock_rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    """
    Best pose per (complex_id, canonical smiles): max CNNaffinity, falling back
    to min minimizedAffinity when the CNN score is absent.
    """
    best: Dict[Tuple[str, str], dict] = {}
    for r in dock_rows:
        if r.get("status") != "ok":
            continue
        key = (r["complex_id"], canonical(r.get("smiles", "")))
        if not key[1]:
            continue
        cur = best.get(key)
        cand_p, cand_s = _f(r.get(PRIMARY)), _f(r.get(SECONDARY))
        if cur is None:
            best[key] = r
            continue
        cur_p, cur_s = _f(cur.get(PRIMARY)), _f(cur.get(SECONDARY))
        if cand_p is not None and (cur_p is None or cand_p > cur_p):
            best[key] = r
        elif cand_p is None and cur_p is None and cand_s is not None and (cur_s is None or cand_s < cur_s):
            best[key] = r
    return best


def parent_reference(dock_rows: List[dict]) -> Dict[str, dict]:
    """Best parent_redock pose per complex, plus its RMSD and crystal score."""
    out: Dict[str, dict] = {}
    for r in dock_rows:
        if r.get("arm") != bc.PARENT_ARM or r.get("status") != "ok":
            continue
        cid = r["complex_id"]
        cur = out.get(cid)
        if cur is None or (_f(r.get(PRIMARY)) or -1e9) > (_f(cur.get(PRIMARY)) or -1e9):
            keep_rmsd = _f(cur.get("redock_rmsd")) if cur else None
            out[cid] = dict(r)
            if out[cid].get("redock_rmsd") in ("", None) and keep_rmsd is not None:
                out[cid]["redock_rmsd"] = keep_rmsd
    for r in dock_rows:                       # RMSD lives on the pose-1 row
        if r.get("arm") == bc.PARENT_ARM and str(r.get("pose")) == "1":
            v = _f(r.get("redock_rmsd"))
            if v is not None and r["complex_id"] in out:
                out[r["complex_id"]]["redock_rmsd"] = v
    return out


def build_per_molecule(preds: List[dict], dock_rows: List[dict]) -> List[dict]:
    best = best_pose_index(dock_rows)
    ref = parent_reference(dock_rows)
    crystal = {r["complex_id"]: r for r in dock_rows
               if r.get("arm") == "parent_crystal" and r.get("status") == "ok"}

    rows: List[dict] = []
    for p in preds:
        cid = p["complex_id"]
        gen = (p.get("generated_smiles") or "").strip()
        cgen = canonical(gen)
        d = best.get((cid, cgen))
        rp = ref.get(cid)
        prim = _f(d.get(PRIMARY)) if d else None
        sec = _f(d.get(SECONDARY)) if d else None
        ref_prim = _f(rp.get(PRIMARY)) if rp else None
        ref_sec = _f(rp.get(SECONDARY)) if rp else None
        cry = crystal.get(cid)

        row = dict(p)
        row.update({
            "canonical_smiles": cgen,
            "scaffold": murcko(gen),
            "dock_CNNaffinity": prim if prim is not None else "",
            "dock_CNNscore": _f(d.get("CNNscore")) if d else "",
            "dock_minimizedAffinity": sec if sec is not None else "",
            "parent_CNNaffinity": ref_prim if ref_prim is not None else "",
            "parent_minimizedAffinity": ref_sec if ref_sec is not None else "",
            "crystal_CNNaffinity": _f(cry.get(PRIMARY)) if cry else "",
            "redock_rmsd": _f(rp.get("redock_rmsd")) if rp else "",
            # Positive = the prediction beat the reference, for BOTH metrics.
            "delta_CNNaffinity": (round(prim - ref_prim, 4)
                                  if (prim is not None and ref_prim is not None) else ""),
            "improvement_vina": (round(ref_sec - sec, 4)
                                 if (sec is not None and ref_sec is not None) else ""),
        })
        row["better_than_parent"] = (1 if isinstance(row["delta_CNNaffinity"], float)
                                     and row["delta_CNNaffinity"] > 0 else
                                     (0 if isinstance(row["delta_CNNaffinity"], float) else ""))
        rows.append(row)
    return rows


def per_complex_deltas(per_mol: List[dict], metric: str = "delta_CNNaffinity",
                       ) -> Dict[str, Dict[str, float]]:
    """{arm: {complex_id: median delta over that arm's valid predictions}}."""
    acc: Dict[str, Dict[str, List[float]]] = {a: {} for a in bc.ARMS}
    for r in per_mol:
        v = _f(r.get(metric))
        if v is None or r["arm"] not in acc:
            continue
        acc[r["arm"]].setdefault(r["complex_id"], []).append(v)
    return {a: {c: median(vs) for c, vs in d.items()} for a, d in acc.items()}


def trustworthy_complexes(per_mol: List[dict], cutoff: float) -> Tuple[set, set]:
    """(complexes within the redock-RMSD cutoff, complexes flagged)."""
    good, bad = set(), set()
    for r in per_mol:
        v = _f(r.get("redock_rmsd"))
        if v is None:
            continue
        (good if v <= cutoff else bad).add(r["complex_id"])
    return good - bad, bad


# ════════════════════════════════════════════════════════════════════════════
#  ARM SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def summarise_arm(rows: List[dict], arm: str, deltas: Dict[str, float]) -> dict:
    valid = [r for r in rows if str(r.get("valid")) in ("1", "1.0")]
    smis = [r["canonical_smiles"] for r in valid if r.get("canonical_smiles")]
    scaf = [r["scaffold"] for r in valid if r.get("scaffold")]
    div, n_fp = mean_pairwise_diversity(smis)

    def col(name, src=None):
        return [v for v in (_f(r.get(name)) for r in (src or valid)) if v is not None]

    d_list = [v for v in deltas.values() if v is not None]
    lo, hi = bootstrap_median_ci(d_list)
    st, p, n_nz = wilcoxon_vs_zero(d_list)
    per_mol_delta = col("delta_CNNaffinity")
    better = [r for r in valid if r.get("better_than_parent") == 1]
    scored = [r for r in valid if isinstance(_f(r.get("delta_CNNaffinity")), float)]
    q1, q3 = iqr(per_mol_delta)

    tox = col("tox21")
    return {
        "arm": arm,
        "label": bc.ARM_LABEL[arm],
        "n_generated": len(rows),
        "n_valid": len(valid),
        "validity_pct": round(100 * len(valid) / len(rows), 1) if rows else None,
        "n_identical_to_parent": sum(1 for r in rows if str(r.get("identical_to_parent")) == "1"),
        "identical_pct": round(100 * sum(1 for r in rows if str(r.get("identical_to_parent")) == "1")
                               / len(rows), 1) if rows else None,
        "mean_mask_tokens": round(sum(col("n_mask_tokens", rows)) / max(1, len(col("n_mask_tokens", rows))), 2)
                            if col("n_mask_tokens", rows) else None,
        # docking
        "n_docked": len(scored),
        "n_complexes_with_delta": len(d_list),
        "median_delta_CNNaffinity": median(d_list),
        "delta_ci95_low": lo, "delta_ci95_high": hi,
        "per_mol_median_delta": median(per_mol_delta),
        "per_mol_delta_q1": q1, "per_mol_delta_q3": q3,
        "pct_better_than_parent": round(100 * len(better) / len(scored), 1) if scored else None,
        "median_improvement_vina": median(col("improvement_vina")),
        "wilcoxon_stat": st, "wilcoxon_p": p, "wilcoxon_n": n_nz,
        # diversity
        "n_unique_smiles": len(set(smis)),
        "unique_smiles_pct": round(100 * len(set(smis)) / len(smis), 1) if smis else None,
        "n_unique_scaffolds": len(set(scaf)),
        "mean_pairwise_diversity": div, "diversity_n": n_fp,
        "median_novelty": median(col("novelty")),
        # toxicity
        "pains_pct": round(100 * sum(col("pains")) / len(col("pains")), 1) if col("pains") else None,
        "brenk_pct": round(100 * sum(col("brenk")) / len(col("brenk")), 1) if col("brenk") else None,
        "any_alert_pct": round(100 * sum(col("any_alert")) / len(col("any_alert")), 1) if col("any_alert") else None,
        "median_n_alerts": median(col("n_alerts")),
        "mean_tox21": round(sum(tox) / len(tox), 3) if tox else None,
        # drug-likeness
        "median_qed": median(col("qed")),
        "median_sa": median(col("sa_raw")),
        "median_mw": median(col("mw")),
        "median_logp": median(col("logp")),
        "lipinski_pass_pct": (round(100 * sum(1 for v in col("lipinski_violations") if v <= 1)
                                    / len(col("lipinski_violations")), 1)
                              if col("lipinski_violations") else None),
    }


def summarise_parents(per_mol: List[dict], dock_rows: List[dict]) -> dict:
    """The parent arm, for the before/after comparison every number needs."""
    seen, parents = set(), []
    for r in per_mol:
        if r["complex_id"] in seen:
            continue
        seen.add(r["complex_id"])
        parents.append(r)
    ref = parent_reference(dock_rows)
    smis = [canonical(r.get("parent_smiles", "")) for r in parents]
    smis = [s for s in smis if s]
    div, n_fp = mean_pairwise_diversity(smis)
    prim = [v for v in (_f(ref[c].get(PRIMARY)) for c in ref) if v is not None]
    rmsd = [v for v in (_f(ref[c].get("redock_rmsd")) for c in ref) if v is not None]
    return {
        "arm": "parent", "label": bc.ARM_LABEL["parent"],
        "n_generated": len(parents), "n_valid": len(parents),
        "n_unique_smiles": len(set(smis)), "n_unique_scaffolds": len({murcko(s) for s in smis} - {""}),
        "mean_pairwise_diversity": div, "diversity_n": n_fp,
        "median_parent_CNNaffinity": median(prim),
        "n_redocked": len(prim),
        "median_redock_rmsd": median(rmsd),
        "redock_within_2A": sum(1 for v in rmsd if v <= 2.0), "redock_n": len(rmsd),
    }


# ════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ════════════════════════════════════════════════════════════════════════════

def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(True, axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)


def make_figures(per_mol: List[dict], summaries: List[dict], parent_sum: dict,
                 deltas: Dict[str, Dict[str, float]], fig_dir: str,
                 flagged: set, have_docking: bool) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    written: List[str] = []
    rng = random.Random(7)
    arms = [a for a in bc.ARMS if any(r["arm"] == a for r in per_mol)]

    def save(fig, name):
        path = os.path.join(fig_dir, name)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        written.append(path)

    # ── Fig 1: docking delta by arm ─────────────────────────────────────────
    if have_docking:
        fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=SURFACE, layout="constrained")
        all_vals = [v for v in (_f(r.get("delta_CNNaffinity")) for r in per_mol) if v is not None]
        if all_vals:      # headroom first, so the per-arm annotations never sit on a point
            lo_v, hi_v = min(all_vals), max(all_vals)
            pad = 0.10 * (hi_v - lo_v or 1)
            ax.set_ylim(lo_v - pad, hi_v + 2.2 * pad)
        for i, arm in enumerate(arms):
            vals = [v for v in (_f(r.get("delta_CNNaffinity")) for r in per_mol
                                if r["arm"] == arm) if v is not None]
            if not vals:
                continue
            bp = ax.boxplot([vals], positions=[i], widths=0.44, patch_artist=True,
                            showfliers=False, medianprops=dict(color=INK, linewidth=2),
                            whiskerprops=dict(color=MUTED, linewidth=1),
                            capprops=dict(color=MUTED, linewidth=1))
            for patch in bp["boxes"]:
                patch.set_facecolor(COLOR[arm])
                patch.set_alpha(0.28)
                patch.set_edgecolor(COLOR[arm])
                patch.set_linewidth(1.5)
            ax.scatter([i + rng.uniform(-0.13, 0.13) for _ in vals], vals, s=22,
                       color=COLOR[arm], alpha=0.75, linewidths=0.8, edgecolors=SURFACE, zorder=3)
            better = 100 * sum(1 for v in vals if v > 0) / len(vals)
            ax.annotate(f"{better:.0f}% better   n={len(vals)}", (i, ax.get_ylim()[1]),
                        xytext=(0, -4), textcoords="offset points",
                        ha="center", va="top", fontsize=8.5, color=INK2)
        ax.axhline(0, color=COLOR["parent"], linestyle="--", linewidth=2, zorder=1)
        ax.annotate("redocked parent (reference)", (-0.45, 0), xytext=(0, -12),
                    textcoords="offset points", ha="left", va="top", fontsize=9,
                    color=COLOR["parent"])
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([bc.ARM_LABEL[a].split(" (")[0] for a in arms])
        _style(ax, "Docking improvement over the redocked parent, by masking strategy",
               "", f"Δ {PRIMARY}  (higher = better than parent)")
        save(fig, "fig1_docking_delta_by_arm.png")

        # ── Fig 2: per-complex paired view ──────────────────────────────────
        cids = sorted({r["complex_id"] for r in per_mol
                       if _f(r.get("parent_CNNaffinity")) is not None},
                      key=lambda c: next((_f(r.get("parent_CNNaffinity")) for r in per_mol
                                          if r["complex_id"] == c), 0) or 0)
        if cids:
            fig, ax = plt.subplots(figsize=(max(7.6, 0.42 * len(cids) + 3), 5.2),
                                   facecolor=SURFACE, layout="constrained")
            for x, cid in enumerate(cids):
                pv = next((_f(r.get("parent_CNNaffinity")) for r in per_mol
                           if r["complex_id"] == cid), None)
                if pv is not None:
                    ax.plot([x - 0.3, x + 0.3], [pv, pv], color=COLOR["parent"],
                            linestyle="--", linewidth=2, zorder=2)
                for arm in arms:
                    vals = [v for v in (_f(r.get("dock_CNNaffinity")) for r in per_mol
                                        if r["complex_id"] == cid and r["arm"] == arm)
                            if v is not None]
                    if vals:
                        ax.scatter([x + rng.uniform(-0.2, 0.2) for _ in vals], vals, s=26,
                                   color=COLOR[arm], alpha=0.8, linewidths=0.8,
                                   edgecolors=SURFACE, zorder=3)
            ax.set_xticks(range(len(cids)))
            ax.set_xticklabels([c.split(":")[0] + ":" + c.split(":")[1] for c in cids],
                               rotation=90, fontsize=7.5)
            handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLOR[a],
                                  label=bc.ARM_LABEL[a].split(" (")[0]) for a in arms]
            handles.append(plt.Line2D([], [], color=COLOR["parent"], linestyle="--",
                                      label="redocked parent"))
            _style(ax, "", "", PRIMARY)
            # AFTER _style: its tick_params(colors=...) would otherwise repaint these.
            for x, cid in enumerate(cids):
                if cid in flagged:
                    ax.get_xticklabels()[x].set_color("#d03b3b")
            # Title and legend both OUTSIDE the axes, each in its own reserved row,
            # so neither can land on a data point or on the other.
            fig.suptitle("Per complex: every prediction against its own redocked parent"
                         + ("   (red label = redock RMSD > cutoff)" if flagged else ""),
                         color=INK, fontsize=11.5, x=0.01, ha="left")
            fig.legend(handles=handles, frameon=False, fontsize=9, ncol=len(handles),
                       loc="outside lower center", labelcolor=INK2)
            save(fig, "fig2_per_complex.png")

    # ── Fig 3: property panels ──────────────────────────────────────────────
    panels = [("qed", "QED (higher = more drug-like)", None),
              ("sa_raw", "Synthetic accessibility (1 easy - 10 hard)", None),
              ("novelty", "Novelty vs parent (1 - Tanimoto)", None),
              ("mw", "Molecular weight (Da)", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.3),
                             facecolor=SURFACE, layout="constrained")
    for ax, (key, title, _) in zip(axes, panels):
        for i, arm in enumerate(arms):
            vals = [v for v in (_f(r.get(key)) for r in per_mol
                                if r["arm"] == arm and str(r.get("valid")) in ("1", "1.0"))
                    if v is not None]
            if not vals:
                continue
            bp = ax.boxplot([vals], positions=[i], widths=0.45, patch_artist=True,
                            showfliers=False, medianprops=dict(color=INK, linewidth=2),
                            whiskerprops=dict(color=MUTED, linewidth=1),
                            capprops=dict(color=MUTED, linewidth=1))
            for patch in bp["boxes"]:
                patch.set_facecolor(COLOR[arm]); patch.set_alpha(0.28)
                patch.set_edgecolor(COLOR[arm]); patch.set_linewidth(1.5)
            ax.scatter([i + rng.uniform(-0.12, 0.12) for _ in vals], vals, s=14,
                       color=COLOR[arm], alpha=0.6, linewidths=0.6, edgecolors=SURFACE, zorder=3)
        pvals = [v for v in (_f(r.get(key)) for r in per_mol) if v is not None] if key == "mw" else []
        parent_vals = []
        seen = set()
        for r in per_mol:
            if r["complex_id"] in seen:
                continue
            seen.add(r["complex_id"])
            pv = _f(r.get(f"parent_{key}"))
            if pv is not None:
                parent_vals.append(pv)
        if parent_vals:
            m = median(parent_vals)
            ax.axhline(m, color=COLOR["parent"], linestyle="--", linewidth=1.8)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([bc.ARM_LABEL[a].split(" (")[0] for a in arms], fontsize=8.5, rotation=20)
        _style(ax, title, "", "")
    fig.suptitle("Property profile of vanilla-ChemBERTa predictions, by masking strategy",
                 color=INK, fontsize=12, x=0.02, ha="left")
    save(fig, "fig3_properties.png")

    # ── Fig 4: validity / diversity / alerts ────────────────────────────────
    metrics = [("validity_pct", "RDKit validity (%)"),
               ("identical_pct", "Identical to parent (%)"),
               ("unique_smiles_pct", "Unique molecules (%)"),
               ("mean_pairwise_diversity", "Mean pairwise diversity (1-Tanimoto)"),
               ("any_alert_pct", "PAINS or Brenk alert (%)"),
               ("lipinski_pass_pct", "Lipinski pass (<=1 violation, %)")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), facecolor=SURFACE, layout="constrained")
    for ax, (key, title) in zip(axes.ravel(), metrics):
        vals, labels, colors = [], [], []
        for arm in arms:
            s = next((d for d in summaries if d["arm"] == arm), None)
            v = s.get(key) if s else None
            vals.append(0 if v is None else v)
            labels.append(bc.ARM_LABEL[arm].split(" (")[0])
            colors.append(COLOR[arm])
        bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:g}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
        if key == "mean_pairwise_diversity" and parent_sum.get("mean_pairwise_diversity"):
            ax.axhline(parent_sum["mean_pairwise_diversity"], color=COLOR["parent"],
                       linestyle="--", linewidth=1.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8.5, rotation=15)
        # Percentages are read against zero, so never let matplotlib crop the base.
        top = max(vals) if max(vals) > 0 else 1
        ax.set_ylim(0, top * 1.22)
        _style(ax, title, "", "")
    fig.suptitle("Validity, diversity, alerts and drug-likeness by masking strategy",
                 color=INK, fontsize=12, x=0.02, ha="left")
    save(fig, "fig4_validity_diversity_alerts.png")

    # ── Fig 5: mask size vs improvement ─────────────────────────────────────
    if have_docking:
        fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE, layout="constrained")
        any_pt = False
        for arm in arms:
            xs, ys = [], []
            for r in per_mol:
                if r["arm"] != arm:
                    continue
                x, y = _f(r.get("n_mask_tokens")), _f(r.get("delta_CNNaffinity"))
                if x is not None and y is not None:
                    xs.append(x + rng.uniform(-0.12, 0.12)); ys.append(y)
            if xs:
                any_pt = True
                ax.scatter(xs, ys, s=26, color=COLOR[arm], alpha=0.72, linewidths=0.8,
                           edgecolors=SURFACE, label=bc.ARM_LABEL[arm].split(" (")[0], zorder=3)
        if any_pt:
            ax.axhline(0, color=COLOR["parent"], linestyle="--", linewidth=2)
            ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
            _style(ax, "Does masking more tokens help? Mask size against docking improvement",
                   "<mask> tokens shown to the model", f"Δ {PRIMARY}")
            save(fig, "fig5_masksize_vs_delta.png")
        else:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
    return written


# ════════════════════════════════════════════════════════════════════════════
#  REPORT
# ════════════════════════════════════════════════════════════════════════════

def _fmt(v, nd=3) -> str:
    if v is None or v == "":
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e4 else f"{v:.3g}"
    return str(v)


def write_report(root: str, summaries: List[dict], parent_sum: dict,
                 deltas: Dict[str, Dict[str, float]], pairwise: List[dict],
                 fried: Tuple, flagged: set, n_complexes: int, figures: List[str],
                 have_docking: bool, cutoff: float, restricted: Optional[dict]) -> str:
    P = bc.paths(root)
    L: List[str] = []
    A = L.append
    A("# Baseline analysis — vanilla ChemBERTa, three masking strategies\n")
    man = {}
    if os.path.exists(P["manifest"]):
        try:
            with open(P["manifest"], encoding="utf-8") as f:
                man = json.load(f)
        except (json.JSONDecodeError, OSError):
            man = {}
    gen = man.get("generation", {})
    masks = man.get("masks", {})
    # The manifest is authoritative when present (it records what actually ran);
    # config is the fallback so a bundle assembled by hand still reports numbers.
    cfg = bc.baseline_settings()
    mset = masks.get("settings", {})
    A(f"**Model:** `{gen.get('model') or cfg['chemberta_model']}` "
      f"(no fine-tuning, no adapter)  \n")
    A(f"**Mask budget:** ≤ {mset.get('mask_percent', cfg['mask_percent'])} % of each parent's "
      f"BPE tokens, identical for all three arms  \n")
    A(f"**Complexes:** {n_complexes}  ·  **predictions per arm per complex:** "
      f"{mset.get('n_seeds', cfg['n_seeds'])}  \n")
    if masks.get("masks_csv_sha256"):
        A(f"**masks.csv sha256:** `{masks['masks_csv_sha256'][:16]}…` "
          f"(re-running another model on the identical masks reproduces this table)\n")
    A("")

    A("## 1. Are the predictions better than the redocked parent?\n")
    if not have_docking:
        A("_No docking results in this bundle yet — run `run_docking.py` on the cluster._\n")
    else:
        A(f"Primary metric **{PRIMARY}** (higher = better). Δ is prediction − redocked parent, "
          f"so **positive = the prediction beat the reference**. Each complex contributes one "
          f"median Δ per arm; the Wilcoxon test runs on those per-complex values.\n")
        A("| arm | complexes | median Δ | 95% CI | molecules scored | % better | median Vinardo gain | Wilcoxon p |")
        A("|---|---:|---:|---|---:|---:|---:|---:|")
        for s in summaries:
            ci = (f"[{_fmt(s['delta_ci95_low'])}, {_fmt(s['delta_ci95_high'])}]"
                  if s["delta_ci95_low"] is not None else "n/a")
            A(f"| {s['label']} | {s['n_complexes_with_delta']} | {_fmt(s['median_delta_CNNaffinity'])} "
              f"| {ci} | {s['n_docked']} | {_fmt(s['pct_better_than_parent'],1)}% "
              f"| {_fmt(s['median_improvement_vina'])} | {_fmt(s['wilcoxon_p'],4)} |")
        A("")
        if parent_sum.get("redock_n"):
            A(f"**Docking-setup check:** the parent redock reproduced the crystal pose within "
              f"{cutoff} Å for {parent_sum['redock_within_2A']}/{parent_sum['redock_n']} complexes "
              f"(median RMSD {_fmt(parent_sum['median_redock_rmsd'],2)} Å). ")
            if flagged:
                A(f"{len(flagged)} complex(es) exceeded it and are flagged: "
                  f"{', '.join(sorted(flagged))}.\n")
            else:
                A("No complex was flagged.\n")
        if restricted:
            A(f"Restricted to the {restricted['n']} complexes that passed the RMSD check, the "
              f"median Δ per arm is: "
              + ", ".join(f"**{a}** {_fmt(restricted['median'][a])}" for a in restricted["median"])
              + ".\n")

    A("## 2. Does the masking strategy matter?\n")
    if fried[1] is not None:
        A(f"Friedman across the three arms on the {fried[2]} complexes where all three have a "
          f"value: χ² = {_fmt(fried[0],2)}, **p = {_fmt(fried[1],4)}**.\n")
    else:
        A("_Friedman test not run: fewer than 5 complexes have a value for all three arms._\n")
    if pairwise:
        A("| comparison | complexes | median difference | p | p (Holm) |")
        A("|---|---:|---:|---:|---:|")
        for pw in pairwise:
            A(f"| {pw['arm_a']} vs {pw['arm_b']} | {pw['n']} | {_fmt(pw['median_diff'])} "
              f"| {_fmt(pw['p'],4)} | {_fmt(pw['p_holm'],4)} |")
        A("")
    A("Mask sizes actually shown to the model (the three arms share one budget, so a smaller "
      "pool simply yields a smaller mask):\n")
    A("| arm | mean \\<mask\\> tokens | valid | identical to parent |")
    A("|---|---:|---:|---:|")
    for s in summaries:
        A(f"| {s['label']} | {_fmt(s['mean_mask_tokens'],2)} | {_fmt(s['validity_pct'],1)}% "
          f"| {_fmt(s['identical_pct'],1)}% |")
    A("")

    A("## 3. Are they chemically diverse?\n")
    A("| arm | valid | unique molecules | unique scaffolds | mean pairwise diversity | median novelty vs parent |")
    A("|---|---:|---:|---:|---:|---:|")
    for s in summaries:
        A(f"| {s['label']} | {s['n_valid']} | {s['n_unique_smiles']} ({_fmt(s['unique_smiles_pct'],1)}%) "
          f"| {s['n_unique_scaffolds']} | {_fmt(s['mean_pairwise_diversity'])} "
          f"| {_fmt(s['median_novelty'])} |")
    A(f"\nFor scale, the {parent_sum['n_generated']} parent ligands themselves have "
      f"{parent_sum['n_unique_scaffolds']} distinct scaffolds and a mean pairwise diversity of "
      f"{_fmt(parent_sum['mean_pairwise_diversity'])}.\n")

    A("## 4. Are they toxic?\n")
    tox_on = any(s["mean_tox21"] is not None for s in summaries)
    A("| arm | PAINS | Brenk | any alert | median alerts |"
      + (" mean Tox21 (1 = clean) |" if tox_on else ""))
    A("|---|---:|---:|---:|---:|" + ("---:|" if tox_on else ""))
    for s in summaries:
        A(f"| {s['label']} | {_fmt(s['pains_pct'],1)}% | {_fmt(s['brenk_pct'],1)}% "
          f"| {_fmt(s['any_alert_pct'],1)}% | {_fmt(s['median_n_alerts'],1)} |"
          + (f" {_fmt(s['mean_tox21'])} |" if tox_on else ""))
    if not tox_on:
        A("\n_Tox21 column absent: `config.STAGE9_TOX21_MODEL_DIR` did not point at a trained "
          "checkpoint when `generate_predictions.py` ran. PAINS/Brenk structural alerts are "
          "reported regardless._\n")
    else:
        A("")

    A("## 5. Are they drug-like?\n")
    A("| arm | median QED | median SA | median MW | median logP | Lipinski pass |")
    A("|---|---:|---:|---:|---:|---:|")
    for s in summaries:
        A(f"| {s['label']} | {_fmt(s['median_qed'])} | {_fmt(s['median_sa'],2)} "
          f"| {_fmt(s['median_mw'],1)} | {_fmt(s['median_logp'],2)} "
          f"| {_fmt(s['lipinski_pass_pct'],1)}% |")
    A("")

    if figures:
        A("## Figures\n")
        for f in figures:
            A(f"* `figures/{os.path.basename(f)}`")
        A("")

    A("## Files\n")
    A("| file | contents |")
    A("|---|---|")
    A("| `masks.csv` | every mask, with the exact atom indices drawn (the reproducibility artifact) |")
    A("| `predictions.csv` | one row per (complex, arm, seed): the generated SMILES and its properties |")
    A("| `docking_summary.csv` | every GNINA pose for every ligand, plus the parent redock and crystal score |")
    A("| `per_molecule.csv` | predictions joined to their best pose and their complex's reference |")
    A("| `per_complex_delta.csv` | per (complex, arm) median Δ — the unit the statistics use |")
    A("| `arm_summary.csv` | this report's numbers, machine-readable |")
    A("| `manifest.json` | settings, model, and sha256 of every artifact |")
    A("")

    text = "\n".join(L)
    with open(P["results"], "w", encoding="utf-8") as f:
        f.write(text)
    return P["results"]


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> int:
    root = bc.bundle_dir(args.out_dir)
    P = bc.paths(root)
    if not os.path.exists(P["predictions"]):
        print(f"ERROR: no predictions.csv at {P['predictions']}. Run generate_predictions.py first.")
        return 2

    preds = bc.read_csv_rows(P["predictions"])
    dock_rows = [] if args.no_docking else bc.read_csv_rows(P["docking"])
    have_docking = bool(dock_rows)
    print(f"predictions: {len(preds)} rows   docking: {len(dock_rows)} rows"
          f"{'' if have_docking else '  (no docking yet — property analysis only)'}")

    per_mol = build_per_molecule(preds, dock_rows)
    deltas = per_complex_deltas(per_mol)
    good, flagged = trustworthy_complexes(per_mol, args.rmsd_cutoff)

    summaries = [summarise_arm([r for r in per_mol if r["arm"] == a], a, deltas.get(a, {}))
                 for a in bc.ARMS if any(r["arm"] == a for r in per_mol)]
    parent_sum = summarise_parents(per_mol, dock_rows)
    fried = friedman(deltas)
    pairwise = pairwise_wilcoxon(deltas)

    restricted = None
    if flagged and have_docking:
        restricted = {"n": len(good),
                      "median": {a: median([v for c, v in deltas.get(a, {}).items() if c in good])
                                 for a in bc.ARMS if a in deltas}}

    n_complexes = len({r["complex_id"] for r in per_mol})
    pm_fields = list(dict.fromkeys(list(preds[0].keys()) + [
        "canonical_smiles", "scaffold", "dock_CNNaffinity", "dock_CNNscore",
        "dock_minimizedAffinity", "parent_CNNaffinity", "parent_minimizedAffinity",
        "crystal_CNNaffinity", "redock_rmsd", "delta_CNNaffinity", "improvement_vina",
        "better_than_parent"])) if preds else []
    bc.write_csv_rows(P["per_molecule"], per_mol, pm_fields)

    dl_rows = [{"complex_id": c, "arm": a, "median_delta_CNNaffinity": v,
                "rmsd_flagged": int(c in flagged)}
               for a, d in deltas.items() for c, v in sorted(d.items())]
    bc.write_csv_rows(P["complex_delta"], dl_rows,
                      ["complex_id", "arm", "median_delta_CNNaffinity", "rmsd_flagged"])

    all_keys = list(dict.fromkeys([k for s in summaries for k in s]))
    bc.write_csv_rows(P["arm_summary"], summaries, all_keys)
    long_rows = [{"metric": k, "arm": s["arm"], "value": s.get(k)}
                 for s in summaries for k in all_keys if k not in ("arm", "label")]
    long_rows += [{"metric": k, "arm": "parent", "value": v} for k, v in parent_sum.items()
                  if k not in ("arm", "label")]
    bc.write_csv_rows(P["stats"], long_rows, ["metric", "arm", "value"])

    figures = []
    if not args.no_figures:
        figures = make_figures(per_mol, summaries, parent_sum, deltas, P["figures"],
                               flagged, have_docking)

    report = write_report(root, summaries, parent_sum, deltas, pairwise, fried,
                          flagged, n_complexes, figures, have_docking,
                          args.rmsd_cutoff, restricted)

    bc.write_manifest(root, "analysis", {
        "n_complexes": n_complexes, "n_predictions": len(preds),
        "have_docking": have_docking, "rmsd_cutoff": args.rmsd_cutoff,
        "flagged_complexes": sorted(flagged),
        "friedman_p": fried[1], "primary_metric": PRIMARY,
        "figures": [os.path.basename(f) for f in figures],
    })

    print("\n── headline ──────────────────────────────────────────────────")
    for s in summaries:
        print(f"{s['label']:<44} valid {_fmt(s['validity_pct'],1):>6}%  "
              f"QED {_fmt(s['median_qed'],3):>6}  "
              f"unique {s['n_unique_smiles']:>4}  "
              f"Δdock {_fmt(s['median_delta_CNNaffinity']):>7}  "
              f"better {_fmt(s['pct_better_than_parent'],1):>6}%")
    print(f"\nreport : {report}")
    for f in figures:
        print(f"figure : {f}")
    return 0


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _self_test() -> int:
    import tempfile

    print("analyze.py self-test")

    assert median([3, 1, 2]) == 2 and median([4, 1, 2, 3]) == 2.5 and median([]) is None
    assert _f("") is None and _f("nan") is None and _f("2.5") == 2.5 and _f(None) is None
    print("  ✓ median / permissive float: missing values stay None, never 0")

    lo, hi = bootstrap_median_ci([1.0, 1.2, 0.9, 1.1, 1.05, 0.95, 1.15], n_boot=2000)
    assert lo is not None and lo <= 1.05 <= hi, (lo, hi)
    assert bootstrap_median_ci([1.0]) == (None, None)
    print(f"  ✓ bootstrap CI brackets the median: [{lo}, {hi}]; too-small sample -> (None, None)")

    st, p, n = wilcoxon_vs_zero([1.0, 1.2, 0.9, 1.4, 1.1, 1.3])
    assert p is not None and p < 0.05, (st, p)
    _, p2, _ = wilcoxon_vs_zero([0.1, -0.1, 0.05])
    assert p2 is None
    print(f"  ✓ Wilcoxon vs 0: consistent positives -> p={p:.4f}; n<5 -> None (no fake p-value)")

    div, n_fp = mean_pairwise_diversity(["CCO", "CCO", "CCO"])
    assert div == 0.0, div
    div2, _ = mean_pairwise_diversity(["CCO", "c1ccccc1N", "C1CCCCC1C(=O)O"])
    assert div2 > 0.5, div2
    assert mean_pairwise_diversity(["CCO"])[0] is None
    print(f"  ✓ diversity: identical set -> {div}; unrelated set -> {div2}; single molecule -> None")

    assert murcko("c1ccccc1CCN") and murcko("CCO") == ""
    print("  ✓ Murcko scaffold: ring system extracted; acyclic molecule -> '' (no scaffold)")

    # Best-pose selection and the delta's sign convention.
    dock = [
        {"complex_id": "1A:L:A:1", "arm": "plip_pos", "k": "0", "smiles": "CCN", "pose": "1",
         "CNNaffinity": "6.0", "minimizedAffinity": "-7.0", "status": "ok"},
        {"complex_id": "1A:L:A:1", "arm": "plip_pos", "k": "0", "smiles": "CCN", "pose": "2",
         "CNNaffinity": "7.5", "minimizedAffinity": "-8.5", "status": "ok"},
        {"complex_id": "1A:L:A:1", "arm": bc.PARENT_ARM, "k": "", "smiles": "CCO", "pose": "1",
         "CNNaffinity": "6.5", "minimizedAffinity": "-7.5", "redock_rmsd": "1.2", "status": "ok"},
    ]
    best = best_pose_index(dock)
    assert _f(best[("1A:L:A:1", canonical("CCN"))]["CNNaffinity"]) == 7.5
    print("  ✓ best pose per molecule = max CNNaffinity (7.5 chosen over 6.0), not pose 1")

    preds = [{"complex_id": "1A:L:A:1", "arm": "plip_pos", "k": "0", "valid": "1.0",
              "parent_smiles": "CCO", "generated_smiles": "CCN", "n_mask_tokens": "2",
              "qed": "0.55", "novelty": "0.4", "pains": "0.0", "brenk": "1.0",
              "any_alert": "1.0", "lipinski_violations": "0", "identical_to_parent": "0"}]
    pm = build_per_molecule(preds, dock)
    assert pm[0]["delta_CNNaffinity"] == 1.0, pm[0]["delta_CNNaffinity"]
    assert pm[0]["improvement_vina"] == 1.0, pm[0]["improvement_vina"]
    assert pm[0]["better_than_parent"] == 1
    print("  ✓ delta signs: CNNaffinity 7.5 vs parent 6.5 -> +1.0; Vinardo -8.5 vs -7.5 -> "
          "+1.0 gain (both positive = prediction wins)")

    # A prediction proposed by two arms shares one docking row (joined by SMILES).
    preds2 = preds + [{**preds[0], "arm": "random", "k": "3"}]
    pm2 = build_per_molecule(preds2, dock)
    assert all(r["delta_CNNaffinity"] == 1.0 for r in pm2)
    print("  ✓ join by (complex, canonical SMILES): the same molecule in two arms gets the same score")

    good, flagged = trustworthy_complexes(pm, 2.0)
    assert good == {"1A:L:A:1"} and not flagged
    bad_pm = [{**pm[0], "redock_rmsd": 3.4}]
    assert trustworthy_complexes(bad_pm, 2.0)[1] == {"1A:L:A:1"}
    print("  ✓ RMSD gate: 1.2 A passes, 3.4 A flags the complex for separate reporting")

    d = per_complex_deltas(pm2)
    assert d["plip_pos"]["1A:L:A:1"] == 1.0
    s = summarise_arm([r for r in pm2 if r["arm"] == "plip_pos"], "plip_pos", d["plip_pos"])
    assert s["n_valid"] == 1 and s["pct_better_than_parent"] == 100.0
    assert s["median_qed"] == 0.55 and s["any_alert_pct"] == 100.0
    print("  ✓ arm summary: counts, % better, QED and alert rates computed over the valid subset")

    # Missing docking must not crash the property analysis.
    pm3 = build_per_molecule(preds, [])
    assert pm3[0]["delta_CNNaffinity"] == "" and pm3[0]["better_than_parent"] == ""
    s3 = summarise_arm(pm3, "plip_pos", {})
    assert s3["median_delta_CNNaffinity"] is None and s3["median_qed"] == 0.55
    print("  ✓ no docking data: deltas stay empty, property statistics still computed")

    # Holm correction is monotone and never exceeds 1.
    groups = {"plip_pos": {f"c{i}": 1.0 + 0.1 * i for i in range(8)},
              "plip_neg": {f"c{i}": 0.2 * i for i in range(8)},
              "random":   {f"c{i}": 0.5 for i in range(8)}}
    pw = pairwise_wilcoxon(groups)
    ps = [o["p_holm"] for o in pw if o["p_holm"] is not None]
    assert all(0 <= x <= 1 for x in ps), ps
    assert len(pw) == 3
    print(f"  ✓ pairwise Wilcoxon + Holm: {len(pw)} comparisons, corrected p in [0,1]")
    fr = friedman(groups)
    assert fr[1] is not None and fr[2] == 8
    print(f"  ✓ Friedman across 3 arms on 8 shared complexes: p={fr[1]:.4g}")

    # Figures and report render end to end.
    with tempfile.TemporaryDirectory() as td:
        P = bc.paths(td)
        rows = []
        for ci in range(6):
            for arm in bc.ARMS:
                for k in range(3):
                    rows.append({
                        "complex_id": f"{ci}ABC:LIG:A:1", "arm": arm, "k": k, "valid": "1.0",
                        "parent_smiles": "CCO", "generated_smiles": ["CCN", "CCC", "c1ccccc1O"][k],
                        "n_mask_tokens": 2 + k, "qed": 0.4 + 0.05 * k, "sa_raw": 3.0,
                        "novelty": 0.3, "mw": 200 + k, "pains": 0.0, "brenk": 0.0,
                        "any_alert": 0.0, "lipinski_violations": 0, "identical_to_parent": "0",
                        "n_alerts": 0,
                    })
        dk = []
        for ci in range(6):
            cid = f"{ci}ABC:LIG:A:1"
            dk.append({"complex_id": cid, "arm": bc.PARENT_ARM, "k": "", "smiles": "CCO",
                       "pose": "1", "CNNaffinity": "6.0", "minimizedAffinity": "-7.0",
                       "redock_rmsd": "1.1", "status": "ok"})
            for smi, aff in [("CCN", "6.8"), ("CCC", "5.4"), ("c1ccccc1O", "6.2")]:
                dk.append({"complex_id": cid, "arm": "plip_pos", "k": "0", "smiles": smi,
                           "pose": "1", "CNNaffinity": aff, "minimizedAffinity": "-7.5",
                           "status": "ok"})
        import generate_predictions as gp
        bc.write_csv_rows(P["predictions"], rows, gp.PRED_FIELDS + ["n_alerts"])
        import run_docking as rd
        bc.write_csv_rows(P["docking"], dk, rd.DOCK_FIELDS)

        rc = run(argparse.Namespace(out_dir=td, no_docking=False, no_figures=False,
                                    rmsd_cutoff=2.0))
        assert rc == 0
        for key in ("per_molecule", "complex_delta", "arm_summary", "stats", "results"):
            assert os.path.exists(P[key]), key
        figs = os.listdir(P["figures"])
        assert len(figs) >= 4, figs
        body = open(P["results"], encoding="utf-8").read()
        for heading in ("better than the redocked parent", "chemically diverse",
                        "Are they toxic", "drug-like", "masking strategy matter"):
            assert heading in body, heading
        assert "Tox21 column absent" in body
        print(f"  ✓ end to end: {len(rows)} predictions -> 5 CSVs, {len(figs)} figures, "
              f"RESULTS.md with all five sections")

    print("\nALL SELF-TESTS PASSED")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse the baseline: docking, diversity, tox, drug-likeness.")
    p.add_argument("--out-dir", default=None, help="Bundle root (default config.BASELINE_DIR).")
    p.add_argument("--no-docking", action="store_true", help="Ignore docking_summary.csv.")
    p.add_argument("--no-figures", action="store_true", help="Tables and report only.")
    p.add_argument("--rmsd-cutoff", type=float, default=2.0,
                   help="Redock RMSD above which a complex is flagged (default 2.0 A).")
    p.add_argument("--test", action="store_true", help="Run the self-test and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.test:
        return _self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

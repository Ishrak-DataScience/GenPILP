# -*- coding: utf-8 -*-
"""
stage5_chembl_matching.py
=========================
Stage 5 of the pipeline — ChEMBL BRD4 nearest-neighbour analysis.

Identical in output structure to Stage 4, but instead of the small
BR4_PDB_Data.csv reference set, uses ALL BRD4-tested compounds from
ChEMBL that have a quantitative pChEMBL value (IC50, Ki, Kd, etc.)
above a user-supplied threshold.

ChEMBL target for BRD4: CHEMBL1163125

Per group, produces (same files as Stage 4):
  similarity_score_histogram.png
  nearest_neighbour_frequency.png
  closeness_summary.txt

Additionally saves:
  chembl_brd4_cache.csv   (in STAGE5_DIR root)
      Cached ChEMBL molecules so re-runs are instant.
      Columns: "Ligand Code", "Smiles", "pchembl_value", "standard_type"
      "Ligand Code" = molecule_chembl_id  (e.g. CHEMBL1234)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1.  ChEMBL is queried via `chembl_webresource_client`.
     Install:  pip install chembl_webresource_client
     Only records with a non-null pChEMBL value are fetched (Ambiguity 2
     Option B) — qualitative "active/inactive" flags are excluded.

A2.  pChEMBL threshold asked at runtime (default 5.0 = potency ≤ 10 µM).
     Only molecules with max(pChEMBL) ≥ threshold are kept after
     deduplication so that when the same molecule appears in multiple
     assays its best (highest) pChEMBL value is used.

A3.  After fetch, molecules are deduplicated by canonical SMILES
     (same rule as Stage 4 Ambiguity 5 Option A).
     Display label = molecule_chembl_id (first encountered for that SMILES).

A4.  Cache behaviour (Ambiguity 4 Option B):
       - If cache exists → prompt "Use cached data? [Y/n]" (default Y).
       - If user says no, or cache does not exist → fetch from ChEMBL
         and (over)write the cache.
     The cache CSV uses the same "Ligand Code" / "Smiles" column names
     as BR4_PDB_Data.csv so load_br4_ligands() from Stage 4 can read it
     directly without modification.

A5.  All Stage 4 analysis functions are imported and reused without
     modification:
       find_nearest_br4, build_closeness_summary,
       plot_similarity_histogram, plot_nn_frequency,
       load_pool_for_ligand, load_br4_ligands.

A6.  The min_heavy_atoms filter (Stage 4 Assumption A3) is applied to
     the ChEMBL set identically — molecules below the threshold are
     excluded before nearest-neighbour search.

A7.  If ChEMBL returns a SMILES that RDKit cannot parse, that molecule
     is silently skipped (consistent with Stage 4 Assumption A8).

A8.  Fetching can take 30–120 seconds depending on network speed and
     pChEMBL threshold. A progress indicator is printed during fetch.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1:  pip install chembl_webresource_client
Step 2:  python stage5_chembl_matching.py

Must be run AFTER stage2_molecule_generation.py.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Answer "yes" to the smoke-test prompt when running the file.
The test uses a synthetic ChEMBL-format cache CSV so no network call
is made. Writes to config.STAGE5_DIR/test/ (wiped each run).
"""

import csv
import glob
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
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

# ── Stage 5 is fully self-contained — no import from stage4 required ─────────

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


def _read_smiles_from_txt(path: str) -> List[str]:
    valid = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                smi = line.strip()
                if smi and Chem.MolFromSmiles(smi) is not None:
                    valid.append(smi)
    except Exception:
        pass
    return valid


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _FP_GEN.GetFingerprint(mol)


def load_br4_ligands(csv_path: str, min_heavy_atoms: int = 7) -> List[Tuple[str, str]]:
    df = pd.read_csv(csv_path)
    smiles_col = next((c for c in df.columns if c.strip().lower() in ("smiles", "smile")), None)
    label_col  = next((c for c in df.columns if c.strip().lower() in ("ligand code", "ligand_code")), None)
    seen_smiles: dict = {}
    for _, row in df.iterrows():
        raw_smi = str(row[smiles_col]).strip()
        label   = str(row[label_col]).strip()
        mol = Chem.MolFromSmiles(raw_smi)
        if mol is None:
            continue
        if mol.GetNumHeavyAtoms() < min_heavy_atoms:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen_smiles:
            seen_smiles[canon] = label
    return [(label, canon) for canon, label in seen_smiles.items()]


def find_nearest_br4(
    generated_smiles: List[str],
    br4_ligands: List[Tuple[str, str]],
) -> List[Tuple[str, str, float]]:
    br4_fps = [(label, _fp(smi)) for label, smi in br4_ligands]
    br4_fps = [(label, fp) for label, fp in br4_fps if fp is not None]
    results = []
    for gen_smi in generated_smiles:
        gen_fp = _fp(gen_smi)
        if gen_fp is None:
            continue
        best_label, best_score = None, -1.0
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
        for f in glob.glob(os.path.join(lig_dir, "ia_mask*.txt")):
            pool.extend(_read_smiles_from_txt(f))
    if pool_choice in ("rand", "both"):
        for f in glob.glob(os.path.join(lig_dir, "rand_mask*.txt")):
            pool.extend(_read_smiles_from_txt(f))
    return list(set(pool))


def build_closeness_summary(
    nn_results: List[Tuple[str, str, float]],
    threshold: float,
    lig_id: str,
) -> str:
    n_total = len(nn_results)
    if n_total == 0:
        return f"  No generated molecules could be compared for {lig_id}.\n"
    above    = [(smi, label, score) for smi, label, score in nn_results if score >= threshold]
    n_above  = len(above)
    n_below  = n_total - n_above
    fraction = n_above / n_total if n_total > 0 else 0
    lines    = [f"  Source ligand  : {lig_id}",
                f"  Generated mols : {n_total}",
                f"  Threshold T    : {threshold:.2f}", ""]
    if n_above == 0:
        lines.append(f"  ✗ None of the {n_total} generated molecules have a match with Tanimoto >= {threshold:.2f}.")
        return "\n".join(lines) + "\n"
    label_counts    = Counter(label for _, label, _ in above)
    dominant_label, dominant_count = label_counts.most_common(1)[0]
    dominant_scores = [s for _, l, s in above if l == dominant_label]
    median_sim      = float(np.median(dominant_scores))
    all_same        = (len(label_counts) == 1)
    if all_same and n_above == n_total:
        lines.append(f"  ✓ All {n_total}/{n_total} generated molecules are most similar to \'{dominant_label}\' (T >= {threshold:.2f}, med = {median_sim:.3f}).")
    elif all_same:
        lines.append(f"  ✓ {n_above}/{n_total} ({fraction*100:.0f}%) generated molecules are most similar to \'{dominant_label}\' (T >= {threshold:.2f}, med = {median_sim:.3f}).")
        lines.append(f"    {n_below}/{n_total} have no match above threshold.")
    else:
        lines.append(f"  ✓ {n_above}/{n_total} ({fraction*100:.0f}%) generated molecules have a nearest neighbour with T >= {threshold:.2f}.")
        lines.append(f"    Dominant BR4 ligand: \'{dominant_label}\' ({dominant_count}/{n_above} of those above threshold, median similarity = {median_sim:.3f}).")
        if n_below:
            lines.append(f"    {n_below}/{n_total} have no match above threshold.")
        lines.append("    Full breakdown (above threshold only):")
        for lbl, count in label_counts.most_common():
            lbl_scores = [s for _, l, s in above if l == lbl]
            lbl_med    = float(np.median(lbl_scores))
            lines.append(f"      \'{lbl}\': {count} mol(s), median similarity = {lbl_med:.3f}")
    return "\n".join(lines) + "\n"


def plot_similarity_histogram(
    nn_results: List[Tuple[str, str, float]],
    threshold: float,
    lig_id: str,
    out_path: str,
) -> bool:
    if not nn_results:
        return False
    scores = [score for _, _, score in nn_results]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(scores, bins=np.linspace(0, 1, 21), color="#4FC3F7", edgecolor="black")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold ({threshold})")
    ax.set_title(f"Nearest-Neighbour Similarity — {lig_id}")
    ax.set_xlabel("Tanimoto Similarity")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


# ════════════════════════════════════════════════════════════════════════════
#  PDB LABEL ENRICHMENT
# ════════════════════════════════════════════════════════════════════════════

def load_pdb_label_map(ref_csv_path: str) -> Dict[str, str]:
    """
    Read BR4_PDB_Data.csv and return {Lig_ChEMBL_ID → PDB ID}.

    Used to enrich x-axis labels in plot_nn_frequency:
      "CHEMBL1957266"  →  "CHEMBL1957266 (5HLS)"

    Only entries where Lig_ChEMBL_ID is non-empty are included.
    If a ChEMBL ID maps to multiple PDB entries, the first encountered is used.
    """
    result: Dict[str, str] = {}
    try:
        df = pd.read_csv(ref_csv_path)
        # Column names from BR4_PDB_Data.csv: PDB ID, Lig_ChEMBL_ID
        pdb_col    = next((c for c in df.columns if "pdb" in c.lower() and "id" in c.lower()), None)
        chembl_col = next((c for c in df.columns if "lig_chembl" in c.lower()), None)
        if pdb_col is None or chembl_col is None:
            print(f"  ⚠️  Could not find PDB ID / Lig_ChEMBL_ID columns in {ref_csv_path}")
            return result
        for _, row in df.iterrows():
            cid = str(row[chembl_col]).strip()
            pid = str(row[pdb_col]).strip()
            if cid and cid.lower() not in ("nan", "") and pid and pid.lower() not in ("nan", ""):
                if cid not in result:
                    result[cid] = pid
    except Exception as e:
        print(f"  ⚠️  Could not load PDB label map: {e}")
    return result


def plot_nn_frequency(
    nn_results: List[Tuple[str, str, float]],
    threshold: float,
    lig_id: str,
    out_path: str,
    pdb_label_map: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Stacked bar chart of nearest-neighbour frequencies (above threshold).
    Bars are coloured by similarity score bins, median annotated on top.

    pdb_label_map : {ChEMBL_ID → PDB_ID} from load_pdb_label_map().
                    When supplied, x-axis labels become "CHEMBLXXXXX (PDBID)"
                    for ChEMBL IDs that have a PDB entry.
    """
    above = [(label, score) for smi, label, score in nn_results if score >= threshold]
    if not above:
        return False

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    label_scores: Dict[str, list] = defaultdict(list)
    for label, score in above:
        label_scores[label].append(score)

    raw_labels = sorted(label_scores.keys(),
                        key=lambda k: len(label_scores[k]), reverse=True)

    # Enrich labels: "CHEMBL1957266" → "CHEMBL1957266 (5HLS)" if PDB entry exists
    def _enrich(lbl: str) -> str:
        if pdb_label_map and lbl in pdb_label_map:
            return f"{lbl} ({pdb_label_map[lbl]})"
        return lbl

    display_labels = [_enrich(lbl) for lbl in raw_labels]

    bins = [
        (threshold, 0.6,  "#FFD54F", f"Low ({threshold:.2f} - 0.60)"),
        (0.6,       0.8,  "#81C784", "Medium (0.60 - 0.80)"),
        (0.8,       1.01, "#388E3C", "High (0.80 - 1.00)"),
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(raw_labels) * 1.2), 6))
    x_positions = np.arange(len(raw_labels))

    for idx, (raw_lbl, disp_lbl) in enumerate(zip(raw_labels, display_labels)):
        scores      = label_scores[raw_lbl]
        median_val  = float(np.median(scores))
        total_h     = 0
        for b_min, b_max, color, _ in bins:
            count = sum(1 for s in scores if b_min <= s < b_max)
            if count > 0:
                ax.bar(x_positions[idx], count, bottom=total_h,
                       color=color, edgecolor="black", width=0.6)
                total_h += count
        ax.text(x_positions[idx],
                total_h + max(1, total_h * 0.02),
                f"Med: {median_val:.2f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", rotation=45, color="black")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_ylabel("Count of Generated Molecules")
    ax.set_title(f"Nearest-Neighbour Frequency (T \u2265 {threshold:.2f}) \u2014 {lig_id}")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=l)
                       for _, _, c, l in bins]
    ax.legend(handles=legend_elements, title="Similarity Range")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True

# BRD4 ChEMBL target ID
BRD4_CHEMBL_TARGET = "CHEMBL1163125"


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
        raw = input(
            "  Minimum heavy atoms for ChEMBL molecules (filters fragments) [7]: "
        ).strip()
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
        raw = input(
            "  Tanimoto similarity threshold T for 'close' decision (0.0–1.0) [0.4]: "
        ).strip()
        if raw == "":
            return 0.4
        try:
            val = float(raw)
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
        print("    Please type a number between 0.0 and 1.0.")


def _ask_pchembl_threshold() -> float:
    print("""
  pChEMBL threshold — only keep BRD4 compounds with pChEMBL ≥ this value.
    pChEMBL = -log10(IC50 in molar).  Examples:
      5.0  →  IC50 ≤ 10 µM   (broad, many compounds)  [default]
      6.0  →  IC50 ≤  1 µM   (moderate activity)
      7.0  →  IC50 ≤ 100 nM  (potent compounds only)
""")
    while True:
        raw = input("  pChEMBL threshold [5.0]: ").strip()
        if raw == "":
            return 5.0
        try:
            val = float(raw)
            if val >= 0.0:
                return val
        except ValueError:
            pass
        print("    Please type a non-negative number.")


def ask_stage5_options(
    cache_exists: bool,
) -> Tuple[str, int, float, float, bool]:
    """
    Ask all runtime options.
    Returns (pool_choice, min_heavy_atoms, threshold, pchembl_threshold,
             use_cache).
    """
    print("\n" + "─" * 60)
    print("STAGE 5 OPTIONS")
    print("─" * 60)

    # Cache decision first — determines whether fetch options are shown
    use_cache = False
    if cache_exists:
        print(f"\n  Cached ChEMBL data found at:\n    {config.CHEMBL_CACHE_PATH}")
        use_cache = _ask_yes_no_default(
            "Use cached ChEMBL data (faster)?", default=True
        )

    pchembl_min = 5.0
    if not use_cache:
        pchembl_min = _ask_pchembl_threshold()

    pool      = _ask_pool_choice()
    min_heavy = _ask_min_heavy_atoms()
    threshold = _ask_threshold()

    print(f"\n  Settings confirmed:")
    if use_cache:
        print(f"    ChEMBL data    : cached  ({config.CHEMBL_CACHE_PATH})")
    else:
        print(f"    pChEMBL ≥      : {pchembl_min}")
    print(f"    Pool           : {pool}")
    print(f"    Min heavy atoms: {min_heavy}")
    print(f"    Threshold T    : {threshold}")
    print("─" * 60 + "\n")

    return pool, min_heavy, threshold, pchembl_min, use_cache


# ════════════════════════════════════════════════════════════════════════════
#  CHEMBL FETCH
# ════════════════════════════════════════════════════════════════════════════

def fetch_brd4_from_chembl(pchembl_min: float) -> List[Tuple[str, str, float, str]]:
    """
    Fetch all BRD4 bioactivity records from ChEMBL with pChEMBL ≥ pchembl_min.

    Requires:  pip install chembl_webresource_client

    Returns
    -------
    List of (molecule_chembl_id, canonical_smiles, pchembl_value, standard_type)
    after deduplication by canonical SMILES (best pChEMBL per molecule kept).
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "chembl_webresource_client is not installed.\n"
            "Run:  pip install chembl_webresource_client"
        )

    print(f"  Fetching BRD4 ({BRD4_CHEMBL_TARGET}) bioactivities "
          f"with pChEMBL ≥ {pchembl_min} from ChEMBL ...")
    print("  (This may take 30–120 seconds depending on network speed.)")

    activity = new_client.activity

    t0 = time.time()
    records = activity.filter(
        target_chembl_id  = BRD4_CHEMBL_TARGET,
        pchembl_value__gte = pchembl_min,
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "standard_type",
    ])

    # Materialise — chembl_webresource_client returns a lazy QuerySet
    raw = list(records)
    elapsed = time.time() - t0
    print(f"  Fetched {len(raw)} raw records in {elapsed:.1f}s.")

    # Deduplicate by canonical SMILES, keeping best (highest) pChEMBL per mol
    # key: canonical_smiles → (chembl_id, pchembl_value, standard_type)
    best: Dict[str, Tuple[str, float, str]] = {}

    skipped = 0
    for rec in raw:
        raw_smi    = (rec.get("canonical_smiles") or "").strip()
        chembl_id  = (rec.get("molecule_chembl_id") or "").strip()
        pchembl    = rec.get("pchembl_value")
        std_type   = (rec.get("standard_type") or "").strip()

        if not raw_smi or not chembl_id:
            skipped += 1
            continue

        mol = Chem.MolFromSmiles(raw_smi)
        if mol is None:
            skipped += 1
            continue

        try:
            pchembl_f = float(pchembl)
        except (TypeError, ValueError):
            skipped += 1
            continue

        canon = Chem.MolToSmiles(mol)

        if canon not in best or pchembl_f > best[canon][1]:
            best[canon] = (chembl_id, pchembl_f, std_type)

    if skipped:
        print(f"  ⚠️  {skipped} record(s) skipped "
              "(missing SMILES, unparseable SMILES, or missing pChEMBL).")

    result = [
        (chembl_id, canon, pchembl_f, std_type)
        for canon, (chembl_id, pchembl_f, std_type) in best.items()
    ]
    print(f"  ✅ {len(result)} unique ChEMBL molecules after deduplication.")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  CACHE  READ / WRITE
# ════════════════════════════════════════════════════════════════════════════

def save_chembl_cache(
    molecules: List[Tuple[str, str, float, str]],
    cache_path: str,
) -> None:
    """
    Save fetched ChEMBL molecules to a CSV using the same column names as
    BR4_PDB_Data.csv so load_br4_ligands() can read it without modification.

    Columns written:
        "Ligand Code"   = molecule_chembl_id
        "Smiles"        = canonical_smiles
        "pchembl_value" = best pChEMBL value
        "standard_type" = assay type (IC50, Ki, etc.)
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Ligand Code", "Smiles", "pchembl_value", "standard_type"],
        )
        writer.writeheader()
        for chembl_id, canon, pchembl_f, std_type in molecules:
            writer.writerow({
                "Ligand Code":   chembl_id,
                "Smiles":        canon,
                "pchembl_value": pchembl_f,
                "standard_type": std_type,
            })
    print(f"  💾 Cache saved: {cache_path}  ({len(molecules)} molecules)")


def load_or_fetch_chembl(
    pchembl_min: float,
    cache_path: str,
    use_cache: bool,
) -> List[Tuple[str, str, float, str]]:
    """
    Return ChEMBL BRD4 molecules either from cache or by fetching.

    If use_cache=True and cache exists, reads the CSV.
    Otherwise fetches from ChEMBL and writes the cache.

    Returns List of (chembl_id, canonical_smiles, pchembl_value, std_type).
    """
    if use_cache and os.path.exists(cache_path):
        print(f"  📂 Reading ChEMBL cache: {cache_path}")
        df = pd.read_csv(cache_path)
        result = []
        for _, row in df.iterrows():
            smi   = str(row.get("Smiles", "")).strip()
            cid   = str(row.get("Ligand Code", "")).strip()
            try:
                pval = float(row.get("pchembl_value", 0))
            except (ValueError, TypeError):
                pval = 0.0
            stype = str(row.get("standard_type", "")).strip()
            if smi and cid:
                result.append((cid, smi, pval, stype))
        print(f"  ✅ {len(result)} molecules loaded from cache.")
        return result

    # Fetch from ChEMBL
    molecules = fetch_brd4_from_chembl(pchembl_min)
    save_chembl_cache(molecules, cache_path)
    return molecules


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def _draw_chembl_boxplot(
    data:       list,
    labels:     list,
    tag:        str,
    strategy:   str,
    out_path:   str,
) -> None:
    """Draw and save one ChEMBL summary boxplot for a single strategy."""
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 6),
                           facecolor="#FAFAFA")
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels,
                    medianprops=dict(color="red", linewidth=2))
    color = {"ia": "#1f77b4", "rand": "#d62728"}.get(strategy, "#2ca02c")
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Ligand Group", fontsize=11)
    ax.set_ylabel("Tanimoto Similarity to Nearest ChEMBL Neighbour", fontsize=11)
    ax.set_title(
        f"ChEMBL Nearest-Neighbour Tanimoto — All Ligands  ({tag})\n"
        f"Stage 5 summary  (N={len(labels)} groups, all nn scores)",
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼  {os.path.basename(out_path)}")


def plot_summary_boxplot_stage5(
    all_results: Dict[str, dict],
    stage5_dir:  str,
    pool_choice: str,
) -> None:
    """
    Summary boxplot(s) for Stage 5 — all ligand groups side by side.

    Outputs (in stage5_dir/summary_plot/):
      pool_choice = "ia"   → 1 plot: boxplot_chembl_tanimoto_ia.png
      pool_choice = "rand" → 1 plot: boxplot_chembl_tanimoto_rand.png
      pool_choice = "both" → 2 plots:
                               boxplot_chembl_tanimoto_ia.png
                               boxplot_chembl_tanimoto_rand.png
    """
    out_dir = os.path.join(stage5_dir, "summary_plot")
    os.makedirs(out_dir, exist_ok=True)
    lig_ids = sorted(all_results.keys())

    def _collect(score_key: str, result_subkey: Optional[str] = None):
        """Return (labels, data) for one strategy."""
        labels_, data_ = [], []
        for lid in lig_ids:
            r = all_results[lid]
            sub = r.get(result_subkey, {}) if result_subkey else r
            scores = sub.get(score_key, [])
            if scores:
                labels_.append(_safe(lid))
                data_.append(scores)
        return labels_, data_

    # Determine which plots to produce
    if pool_choice == "ia":
        tasks = [("ia", "IA only", None)]
    elif pool_choice == "rand":
        tasks = [("rand", "Random only", None)]
    else:  # both → 2 plots
        tasks = [
            ("ia",   "IA only",      "ia"),
            ("rand", "Random only",  "rand"),
        ]

    for strategy, tag, subkey in tasks:
        labels, data = _collect("nn_scores", subkey)
        if not data:
            print(f"  ⚠️  Stage 5 summary boxplot ({strategy}): no data. Skipped.")
            continue
        out_path = os.path.join(out_dir, f"boxplot_chembl_tanimoto_{strategy}.png")
        _draw_chembl_boxplot(data, labels, tag, strategy, out_path)


def run_stage5(
    pred_dir:        Optional[str] = None,
    stage5_dir:      Optional[str] = None,
    cache_path:      Optional[str] = None,
    pool_choice:     str   = "both",
    min_heavy_atoms: int   = 7,
    threshold:       float = 0.4,
    pchembl_min:     float = 5.0,
    use_cache:       bool  = True,
    
) -> Dict[str, dict]:
    """
    Run Stage-5 ChEMBL BRD4 nearest-neighbour analysis.

    Fetches (or loads from cache) ChEMBL BRD4 actives, then runs the
    identical nearest-neighbour + plotting pipeline as Stage 4.

    Returns
    -------
    Dict keyed by ligand_id → same structure as stage4.run_stage4()
    """
    pred_dir   = pred_dir
    stage5_dir = stage5_dir or config.STAGE5_DIR
    cache_path = cache_path or config.CHEMBL_CACHE_PATH

    os.makedirs(stage5_dir, exist_ok=True)

    # ── Load PDB label map from BR4_PDB_Data.csv (optional enrichment) ────────
    pdb_label_map = load_pdb_label_map(config.REF_CSV_PATH)
    print(f"  PDB label map loaded: {len(pdb_label_map)} ChEMBL→PDB entries")

    # ── Load ChEMBL molecules ─────────────────────────────────────────────────
    chembl_mols = load_or_fetch_chembl(pchembl_min, cache_path, use_cache)

    if not chembl_mols:
        print("  ❌ ChEMBL molecule list is empty. Exiting.")
        return {}

    # ── Write a temporary CSV so load_br4_ligands() can apply min_heavy filter
    #    (reuses Stage 4 function unchanged — Assumption A5) ───────────────────
    tmp_csv = os.path.join(stage5_dir, "_chembl_tmp.csv")
    save_chembl_cache(chembl_mols, tmp_csv)

    chembl_ligands = load_br4_ligands(tmp_csv, min_heavy_atoms=min_heavy_atoms)

    # Clean up temp file
    try:
        os.remove(tmp_csv)
    except OSError:
        pass

    if not chembl_ligands:
        print("  ❌ ChEMBL reference set is empty after filtering. Exiting.")
        return {}

    print(f"  ChEMBL BRD4 reference set: {len(chembl_ligands)} unique ligands "
          f"(pChEMBL ≥ {pchembl_min}, min_heavy ≥ {min_heavy_atoms})")

    # ── Discover ligand folders in PRED_DIR ───────────────────────────────────
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
        out_dir = os.path.join(stage5_dir, _safe(lig_id))
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Ligand: {lig_id}  |  pool: {pool_choice}")
        print(f"{'='*60}")

        def _run_one_strategy(
            strategy_pool:  list,
            strategy_label: str,
            suffix:         str,
        ) -> dict:
            """
            Run nn search + plots + summary for one strategy pool.
            suffix : "_ia", "_rand", or "" (combined/both).
            Returns result dict.
            """
            if not strategy_pool:
                print(f"  ⚠️  Empty pool for {suffix or 'combined'}. Skipping.")
                return {}

            nn   = find_nearest_br4(strategy_pool, chembl_ligands)
            summ = build_closeness_summary(nn, threshold, lig_id)
            print(f"  [{suffix or 'combined'}] {len(strategy_pool)} molecules, "
                  f"{len(nn)} nn-pairs")
            print(summ)

            s_path = os.path.join(
                out_dir,
                f"closeness_summary_{_safe(lig_id)}{suffix}.txt",
            )
            with open(s_path, "w", encoding="utf-8") as fh:
                fh.write("Stage 5 — ChEMBL BRD4 Nearest-Neighbour Closeness Summary\n")
                fh.write(f"ChEMBL target  : {BRD4_CHEMBL_TARGET}\n")
                fh.write(f"pChEMBL ≥      : {pchembl_min}\n")
                fh.write(f"Pool choice    : {strategy_label}\n")
                fh.write(f"Min heavy atoms: {min_heavy_atoms}\n\n")
                fh.write(summ)
            print(f"  💾 {os.path.basename(s_path)}")

            sim_ok = plot_similarity_histogram(
                nn_results = nn,
                threshold  = threshold,
                lig_id     = lig_id,
                out_path   = os.path.join(
                    out_dir,
                    f"similarity_score_histogram_{_safe(lig_id)}{suffix}.png",
                ),
            )
            freq_ok = plot_nn_frequency(
                nn_results    = nn,
                threshold     = threshold,
                lig_id        = lig_id,
                out_path      = os.path.join(
                    out_dir,
                    f"nearest_neighbour_frequency_{_safe(lig_id)}{suffix}.png",
                ),
                pdb_label_map = pdb_label_map,
            )
            if sim_ok:
                print(f"  🖼  similarity_score_histogram_{_safe(lig_id)}{suffix}.png saved.")
            if freq_ok:
                print(f"  🖼  nearest_neighbour_frequency_{_safe(lig_id)}{suffix}.png saved.")

            return {
                "nn_results":    nn,
                "summary_text":  summ,
                "sim_hist_ok":   sim_ok,
                "freq_chart_ok": freq_ok,
                # Raw Tanimoto scores (all nn scores regardless of threshold)
                "nn_scores": [score for _, _, score in nn],
            }

        # ── Load pools and run per strategy (Option B + Option A) ─────────────
        if pool_choice == "ia":
            ia_pool = load_pool_for_ligand(lig_dir, "ia")
            if not ia_pool:
                print("  ⚠️  Empty IA pool. Skipping.")
                continue
            r = _run_one_strategy(ia_pool, "ia", "_ia")
            all_results[lig_id] = {"n_generated": len(ia_pool), **r}

        elif pool_choice == "rand":
            rand_pool = load_pool_for_ligand(lig_dir, "rand")
            if not rand_pool:
                print("  ⚠️  Empty rand pool. Skipping.")
                continue
            r = _run_one_strategy(rand_pool, "rand", "_rand")
            all_results[lig_id] = {"n_generated": len(rand_pool), **r}

        else:  # both
            # Combined pool (no suffix — Option A: keep existing combined file)
            both_pool = load_pool_for_ligand(lig_dir, "both")
            if not both_pool:
                print("  ⚠️  Empty pool. Skipping.")
                continue
            r_both = _run_one_strategy(both_pool, "both", "")

            # IA-only (suffix _ia)
            ia_pool   = load_pool_for_ligand(lig_dir, "ia")
            r_ia      = _run_one_strategy(ia_pool, "ia", "_ia")

            # Random-only (suffix _rand)
            rand_pool = load_pool_for_ligand(lig_dir, "rand")
            r_rand    = _run_one_strategy(rand_pool, "rand", "_rand")

            all_results[lig_id] = {
                "n_generated": len(both_pool),
                "combined":    r_both,
                "ia":          r_ia,
                "rand":        r_rand,
            }

    # ── Summary boxplot: all ligands side by side ───────────────────────────
    plot_summary_boxplot_stage5(all_results, stage5_dir, pool_choice)

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED TEST  (no network call — uses synthetic cache CSV)
# ════════════════════════════════════════════════════════════════════════════

# Synthetic ChEMBL-format molecules for the test
_TEST_CHEMBL_ROWS = [
    # header
    ("Ligand Code", "Smiles", "pchembl_value", "standard_type"),
    # fragments / solvents — should be filtered at min_heavy=7
    ("CHEMBL001", "O",    "5.0", "IC50"),
    ("CHEMBL002", "OCCO", "5.0", "IC50"),
    # real drug-like BRD4 actives
    ("CHEMBL003", "CC(=O)Oc1ccccc1C(=O)O",        "6.5", "IC50"),  # aspirin-like
    ("CHEMBL004", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   "7.2", "Ki"),    # ibuprofen-like
    ("CHEMBL005", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",   "5.8", "IC50"),  # caffeine-like
    ("CHEMBL006", "c1ccc2ccccc2c1",                "5.1", "IC50"),  # naphthalene
]

_TEST_GENERATED = [
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin — should match CHEMBL003
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen — should match CHEMBL004
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",    # caffeine — should match CHEMBL005
]


def _run_test() -> bool:
    print("\n" + "=" * 60)
    print("STAGE 5 SELF-TEST  (no network call)")
    print("=" * 60)

    # ── FIX: 'out_dir' was completely undefined here.
    # Defaulting test folder to config.STAGE5_DIR/test
    td = os.path.join(config.STAGE5_DIR, "test")
    if os.path.exists(td):
        shutil.rmtree(td)
        print(f"  🗑  Wiped existing test dir: {td}")
    os.makedirs(td)
    print(f"  📁 Created fresh test dir:  {td}\n")

    pred_dir    = os.path.join(td, "preds")
    stage5_dir  = os.path.join(td, "stage5_out")
    cache_path  = os.path.join(td, "test_chembl_cache.csv")

    # ── Write synthetic Stage-2 prediction txt files ──────────────────────────
    lig_id  = "1GH-A-1173"
    lig_dir = os.path.join(pred_dir, _safe(lig_id))
    os.makedirs(lig_dir)
    for mc in range(1, 4):
        with open(os.path.join(lig_dir, f"ia_mask{mc:03d}.txt"), "w") as f:
            f.write("\n".join(_TEST_GENERATED) + "\n")
        with open(os.path.join(lig_dir, f"rand_mask{mc:03d}.txt"), "w") as f:
            f.write("\n".join(_TEST_GENERATED[:2]) + "\n")

    # ── Write synthetic ChEMBL cache CSV ─────────────────────────────────────
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in _TEST_CHEMBL_ROWS:
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

    # ── Run Stage 5 with use_cache=True (reads the synthetic CSV) ─────────────
    results = run_stage5(
        pred_dir        = pred_dir,
        stage5_dir      = stage5_dir,        # <── FIX: Changed 'out_dir' to 'stage5_dir'
        cache_path      = cache_path,
        pool_choice     = "both",
        min_heavy_atoms = 7,
        threshold       = 0.4,
        pchembl_min     = 5.0,
        use_cache       = True,
    )
    check(lig_id in results,
          f"'{lig_id}' present in results")

    if lig_id in results:
        r = results[lig_id]

        check(r["n_generated"] > 0,
              f"pool non-empty (got {r['n_generated']})")
        check(len(r["nn_results"]) > 0,
              "nn_results non-empty")
        check(r["sim_hist_ok"],
              "similarity_score_histogram_{lig_id_safe}.png produced")
        check(r["freq_chart_ok"],
              "nearest_neighbour_frequency_{lig_id_safe}.png produced")

        # Solvents must be filtered
        labels_used = {label for _, label, _ in r["nn_results"]}
        check("CHEMBL001" not in labels_used and "CHEMBL002" not in labels_used,
              "solvents CHEMBL001/CHEMBL002 filtered out (min_heavy=7)")

        # Scores in [0, 1]
        scores = [s for _, _, s in r["nn_results"]]
        check(all(0.0 <= s <= 1.0 for s in scores),
              "all similarity scores in [0, 1]")

        # Output files on disk
        out_dir = os.path.join(stage5_dir, _safe(lig_id))
        lig_id_safe = _safe(lig_id)
        for fname in [f"similarity_score_histogram_{lig_id_safe}.png",
                      f"nearest_neighbour_frequency_{lig_id_safe}.png",
                      f"closeness_summary_{lig_id_safe}.txt"]:
            check(os.path.exists(os.path.join(out_dir, fname)),
                  f"{fname} on disk")

        # Aspirin should map to CHEMBL003 (identical SMILES)
        asp_smi = "CC(=O)Oc1ccccc1C(=O)O"
        asp_nn = [(label, score) for gen, label, score in r["nn_results"]
                  if gen == asp_smi]
        if asp_nn:
            check(asp_nn[0][0] == "CHEMBL003",
                  f"aspirin → nearest neighbour is CHEMBL003 "
                  f"(got '{asp_nn[0][0]}')")
        else:
            check(False, "aspirin molecule found in nn_results")

    # ── Verify cache-read path works (use_cache=True, file exists) ────────────
    stage5_dir_b = os.path.join(td, "stage5_out_b")
    results_b = run_stage5(
        pred_dir        = pred_dir,
        stage5_dir      = stage5_dir_b,
        cache_path      = cache_path,
        pool_choice     = "ia",
        min_heavy_atoms = 7,
        threshold       = 0.4,
        pchembl_min     = 5.0,
        use_cache       = True,
    )
    check(lig_id in results_b,
          "cache re-use run also produces results")

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
    print("STAGE 5: ChEMBL BRD4 NEAREST-NEIGHBOUR MATCHING")
    print("=" * 60)

    cache_exists = os.path.exists(config.CHEMBL_CACHE_PATH)

  

    run_test = _ask_yes_no_default("Run the smoke test?", default=False)
    if run_test:
        ok = _run_test()
        sys.exit(0 if ok else 1)

    pool, min_heavy, threshold, pchembl_min, use_cache = ask_stage5_options(
        cache_exists=cache_exists
    )
    # 2. Add prompt for the Data Source (Stage 2, 2.5, or 2.7)
    print("""
  Option - Data Source:
    Which dataset are we analyzing?
    [1] Standard Stage 2
    [2] Stage 2.5 (Random Pick)
    [3] Stage 2.7 (Multi-seed)
""")
    while True:
        choice = input("  Enter 1, 2, or 3 [Default: 1]: ").strip()
        if choice in ['1', '']:
            stage_choice = '2'
            in_dir = config.PRED_DIR
            break
        elif choice == '2':
            stage_choice = '2.5'
            in_dir = config.STAGE25_PRED_DIR
            break
        elif choice == '3':
            stage_choice = '2.7'
            in_dir = config.STAGE27_PRED_DIR
            break
        else:
            print("  ❌ Invalid choice. Please enter 1, 2, or 3.")
  
   
    
    out_dir = os.path.join(
        config.STAGE5_DIR, 
        f"generator_Stage{stage_choice}", 
        str(pool) # Converts your pool choice (e.g., "IA", "Random", "Combined") to a folder name
    )
    # 4. Create the nested directories if they don't exist
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  [Data routing: Reading from {in_dir}]")
    print(f"  [Data routing: Saving to {out_dir}]\n")

    # 5. Run the stage with dynamic paths
    results = run_stage5(
        pred_dir        = in_dir,
        stage5_dir      = out_dir,      # <── FIX: Changed 'out_dir' to 'stage5_dir'
        pool_choice     = pool,
        min_heavy_atoms = min_heavy,
        threshold       = threshold,
        pchembl_min     = pchembl_min,
        use_cache       = use_cache,
    )
    print(f"""
  This stage fetches all BRD4-active compounds from ChEMBL
  (target: {BRD4_CHEMBL_TARGET}) and finds the most similar ChEMBL
  molecule for each generated molecule — using the same analysis as
  Stage 4 but on a much larger reference set.

  Cache path: {config.CHEMBL_CACHE_PATH}
  Cache exists: {"YES — will offer to reuse" if cache_exists else "NO — will fetch from ChEMBL"}

  Before running the full analysis you can run a smoke test instead.
  The smoke test uses a synthetic cache CSV — NO network call is made.
   Writes to: {os.path.join(out_dir, "test")}
  (wiped clean at the start of every test run).
""")
    print("\n" + "=" * 60)
    print("✅ Stage 5 complete.")
    print(f"   Outputs saved to: {out_dir}")
    print(f"   Ligands analysed: {len(results)}")
    for lig_id, r in sorted(results.items()):
        n = r.get("n_generated", 0)
        # Result structure differs by pool_choice:
        #   ia/rand  → flat: r["sim_hist_ok"], r["freq_chart_ok"]
        #   both     → nested: r["combined"]["sim_hist_ok"], r["ia"][...], r["rand"][...]
        if "combined" in r:
            c = r["combined"]
            sim_ok  = c.get("sim_hist_ok",   False)
            freq_ok = c.get("freq_chart_ok", False)
            ia_ok   = r.get("ia",   {}).get("sim_hist_ok", False)
            rnd_ok  = r.get("rand", {}).get("sim_hist_ok", False)
            print(
                f"   • {lig_id}  n={n}  "
                f"combined={'✓' if sim_ok else '✗'}  "
                f"ia={'✓' if ia_ok else '✗'}  "
                f"rand={'✓' if rnd_ok else '✗'}"
            )
        else:
            print(
                f"   • {lig_id}  n={n}  "
                f"sim_hist={'✓' if r.get('sim_hist_ok') else '✗'}  "
                f"freq_chart={'✓' if r.get('freq_chart_ok') else '✗'}"
            )


if __name__ == "__main__":
    main()
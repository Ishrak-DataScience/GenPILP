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
"""

import csv
import glob
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem

import config

# ── Reuse ALL analysis functions from Stage 4 (v2) ───────────────────────────
from stage4_br4_matching import (
    build_closeness_summary,
    find_nearest_br4,
    load_br4_ligands,
    load_pool_for_ligand,
    plot_nn_frequency,
    plot_similarity_histogram,
    plot_nn_distribution_boxplot,   # <-- NEW Boxplot import
    _safe,
)

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
        raw = input("  Minimum heavy atoms for ChEMBL molecules (filters fragments) [7]: ").strip()
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
        raw = input("  Tanimoto similarity threshold T for 'close' decision (0.0–1.0) [0.4]: ").strip()
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

def ask_stage5_options(cache_exists: bool) -> Tuple[str, int, float, float, bool]:
    print("\n" + "─" * 60)
    print("STAGE 5 OPTIONS")
    print("─" * 60)

    use_cache = False
    if cache_exists:
        print(f"\n  Cached ChEMBL data found at:\n    {config.CHEMBL_CACHE_PATH}")
        use_cache = _ask_yes_no_default("Use cached ChEMBL data (faster)?", default=True)

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
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError("Run: pip install chembl_webresource_client")

    print(f"  Fetching BRD4 ({BRD4_CHEMBL_TARGET}) bioactivities with pChEMBL ≥ {pchembl_min} from ChEMBL ...")
    activity = new_client.activity
    records = activity.filter(
        target_chembl_id=BRD4_CHEMBL_TARGET, pchembl_value__gte=pchembl_min
    ).only(["molecule_chembl_id", "canonical_smiles", "pchembl_value", "standard_type"])

    raw = list(records)
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

    result = [(chembl_id, canon, pchembl_f, std_type) for canon, (chembl_id, pchembl_f, std_type) in best.items()]
    return result

def save_chembl_cache(molecules: List[Tuple[str, str, float, str]], cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Ligand Code", "Smiles", "pchembl_value", "standard_type"])
        writer.writeheader()
        for chembl_id, canon, pchembl_f, std_type in molecules:
            writer.writerow({"Ligand Code": chembl_id, "Smiles": canon, "pchembl_value": pchembl_f, "standard_type": std_type})
    print(f"  💾 Cache saved: {cache_path}  ({len(molecules)} molecules)")

def load_or_fetch_chembl(pchembl_min: float, cache_path: str, use_cache: bool):
    if use_cache and os.path.exists(cache_path):
        print(f"  📂 Reading ChEMBL cache: {cache_path}")
        df = pd.read_csv(cache_path)
        result = []
        for _, row in df.iterrows():
            smi   = str(row.get("Smiles", "")).strip()
            cid   = str(row.get("Ligand Code", "")).strip()
            try: pval = float(row.get("pchembl_value", 0))
            except (ValueError, TypeError): pval = 0.0
            stype = str(row.get("standard_type", "")).strip()
            if smi and cid:
                result.append((cid, smi, pval, stype))
        print(f"  ✅ {len(result)} molecules loaded from cache.")
        return result

    molecules = fetch_brd4_from_chembl(pchembl_min)
    save_chembl_cache(molecules, cache_path)
    return molecules

# ════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION
# ════════════════════════════════════════════════════════════════════════════

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
    
    stage5_dir = stage5_dir or config.STAGE5_DIR 
    cache_path = cache_path or config.CHEMBL_CACHE_PATH

    os.makedirs(stage5_dir, exist_ok=True)
    chembl_mols = load_or_fetch_chembl(pchembl_min, cache_path, use_cache)

    if not chembl_mols:
        print("  ❌ ChEMBL molecule list is empty. Exiting.")
        return {}

    tmp_csv = os.path.join(stage5_dir, "_chembl_tmp.csv")
    save_chembl_cache(chembl_mols, tmp_csv)

    chembl_ligands = load_br4_ligands(tmp_csv, min_heavy_atoms=min_heavy_atoms)
    try: os.remove(tmp_csv)
    except OSError: pass

    if not chembl_ligands:
        print("  ❌ ChEMBL reference set is empty after filtering. Exiting.")
        return {}

    print(f"  ChEMBL BRD4 reference set: {len(chembl_ligands)} unique ligands (pChEMBL ≥ {pchembl_min}, min_heavy ≥ {min_heavy_atoms})")

    lig_dirs = sorted([d for d in glob.glob(os.path.join(pred_dir, "*")) if os.path.isdir(d)])
    if not lig_dirs:
        print(f"  ❌ No ligand sub-folders found in {pred_dir}")
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

        pool = load_pool_for_ligand(lig_dir, pool_choice)
        print(f"  Generated molecules in pool: {len(pool)}")

        if not pool:
            print("  ⚠️  Empty pool. Skipping.")
            continue

        nn_results = find_nearest_br4(pool, chembl_ligands)
        print(f"  Nearest-neighbour pairs computed: {len(nn_results)}")

        summary = build_closeness_summary(nn_results, threshold, lig_id)
        print(summary)

        summary_path = os.path.join(out_dir, "closeness_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"  💾 Summary saved: {summary_path}")

        sim_ok = plot_similarity_histogram(
            nn_results = nn_results, threshold = threshold, lig_id = lig_id,
            out_path   = os.path.join(out_dir, "similarity_score_histogram.png"),
        )
        freq_ok = plot_nn_frequency(
            nn_results = nn_results, threshold = threshold, lig_id = lig_id,
            out_path   = os.path.join(out_dir, "nearest_neighbour_frequency.png"),
        )
        
        # ---> TRIGGER THE NEW BOXPLOT SAVING HERE
        box_ok = plot_nn_distribution_boxplot(
            nn_results = nn_results, threshold = threshold, lig_id = lig_id,
            out_path   = os.path.join(out_dir, "nearest_neighbour_distribution.png"),
        )

        if sim_ok:  print(f"  🖼  similarity_score_histogram.png saved.")
        if freq_ok: print(f"  🖼  nearest_neighbour_frequency.png saved.")
        if box_ok:  print(f"  🖼  nearest_neighbour_distribution.png saved.")

        all_results[lig_id] = {
            "n_generated":    len(pool),
            "nn_results":     nn_results,
            "summary_text":   summary,
            "sim_hist_ok":    sim_ok,
            "freq_chart_ok":  freq_ok,
            "box_chart_ok":   box_ok,  # Let main() know the boxplot succeeded
        }

    return all_results

# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("STAGE 5: ChEMBL BRD4 NEAREST-NEIGHBOUR MATCHING")
    print("=" * 60)

    cache_exists = os.path.exists(config.CHEMBL_CACHE_PATH)
    
    # 1. Ask Options
    pool, min_heavy, threshold, pchembl_min, use_cache = ask_stage5_options(cache_exists=cache_exists)
    
    # 2. Add prompt for the Data Source
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
        str(pool)
    )
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n  [Data routing: Reading from {in_dir}]")
    print(f"  [Data routing: Saving to {out_dir}]\n")

    # 5. Run the stage with dynamic paths
    results = run_stage5(
        pred_dir        = in_dir,
        stage5_dir      = out_dir,
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
        # Log the 3 success markers back to the user
        print(
            f"   • {lig_id}  n={r['n_generated']}  "
            f"sim_hist={'✓' if r['sim_hist_ok'] else '✗'}  "
            f"freq_chart={'✓' if r['freq_chart_ok'] else '✗'}  "
            f"box_chart={'✓' if r.get('box_chart_ok') else '✗'}"
        )

if __name__ == "__main__":
    main()
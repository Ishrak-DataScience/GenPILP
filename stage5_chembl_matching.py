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
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem

import config

# ── Reuse ALL analysis functions from Stage 4 unchanged ──────────────────────
from stage4_br4_matching import (
    build_closeness_summary,
    find_nearest_br4,
    load_br4_ligands,       # reads cache CSV using "Ligand Code" / "Smiles" columns
    load_pool_for_ligand,
    plot_nn_frequency,
    plot_similarity_histogram,
    _safe,
)

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
    # ── FIX: removed 'out_dir' scope error, defaults to config.STAGE5_DIR
    stage5_dir = stage5_dir or config.STAGE5_DIR 
    cache_path = cache_path or config.CHEMBL_CACHE_PATH

    os.makedirs(stage5_dir, exist_ok=True)

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

        # Load generated pool (reuses Stage 4 function — Assumption A5)
        pool = load_pool_for_ligand(lig_dir, pool_choice)
        print(f"  Generated molecules in pool: {len(pool)}")

        if not pool:
            print("  ⚠️  Empty pool. Skipping.")
            continue

        # Nearest-neighbour search (reuses Stage 4 — Assumption A5)
        nn_results = find_nearest_br4(pool, chembl_ligands)
        print(f"  Nearest-neighbour pairs computed: {len(nn_results)}")

        # Closeness summary (reuses Stage 4 — Assumption A5)
        summary = build_closeness_summary(nn_results, threshold, lig_id)
        print(summary)

        summary_path = os.path.join(out_dir, "closeness_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Stage 5 — ChEMBL BRD4 Nearest-Neighbour Closeness Summary\n")
            f.write(f"ChEMBL target  : {BRD4_CHEMBL_TARGET}\n")
            f.write(f"pChEMBL ≥      : {pchembl_min}\n")
            f.write(f"Pool choice    : {pool_choice}\n")
            f.write(f"Min heavy atoms: {min_heavy_atoms}\n\n")
            f.write(summary)
        print(f"  💾 Summary saved: {summary_path}")

        # Plots (reuse Stage 4 — Assumption A5)
        sim_ok = plot_similarity_histogram(
            nn_results = nn_results,
            threshold  = threshold,
            lig_id     = lig_id,
            out_path   = os.path.join(out_dir, "similarity_score_histogram.png"),
        )
        freq_ok = plot_nn_frequency(
            nn_results = nn_results,
            threshold  = threshold,
            lig_id     = lig_id,
            out_path   = os.path.join(out_dir, "nearest_neighbour_frequency.png"),
        )
        if sim_ok:
            print(f"  🖼  similarity_score_histogram.png saved.")
        if freq_ok:
            print(f"  🖼  nearest_neighbour_frequency.png saved.")

        all_results[lig_id] = {
            "n_generated":    len(pool),
            "nn_results":     nn_results,
            "summary_text":   summary,
            "sim_hist_ok":    sim_ok,
            "freq_chart_ok":  freq_ok,
        }

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
              "similarity_score_histogram.png produced")
        check(r["freq_chart_ok"],
              "nearest_neighbour_frequency.png produced")

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
        for fname in ["similarity_score_histogram.png",
                      "nearest_neighbour_frequency.png",
                      "closeness_summary.txt"]:
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
        print(
            f"   • {lig_id}  n={r['n_generated']}  "
            f"sim_hist={'✓' if r['sim_hist_ok'] else '✗'}  "
            f"freq_chart={'✓' if r['freq_chart_ok'] else '✗'}"
        )


if __name__ == "__main__":
    main()
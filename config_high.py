# -*- coding: utf-8 -*-
"""
py  ─  Shareduration for all pipeline stages.

▶ Edit BASE_DIR and USER_PREFIX, then all output paths update automatically.
"""

# ── Root directory ─────────────────────────────────────────────────────────────
BASE_DIR    = "/content/drive/MyDrive/GenAI4Drug"
USER_PREFIX = "Ishrak"
EXPERIMENT_TAG = "expo05"

# ── All outputs live under BASE_DIR / USER_PREFIX / Output / <subdir> ────────── old
#_OUT = f"{BASE_DIR}/{USER_PREFIX}/Output"

# ── All outputs live under BASE_DIR / USER_PREFIX / Output / <subdir> ──────────New way
_OUT = f"{BASE_DIR}/{USER_PREFIX}/Output/{EXPERIMENT_TAG}"

# ── Stage 1 inputs (not user-specific; shared raw data) ───────────────────────
BASE_PDB_PATH = f"{BASE_DIR}/Dummy_data/PDB/"
BASE_XML_PATH = f"{BASE_DIR}/Dummy_data/plip/"
REF_CSV_PATH  = f"{BASE_DIR}/Dummy_data/BR4_PDB_Data.csv"

# ── Output directories ─────────────────────────────────────────────────────────
MASK_CALC_OUTDIR   = f"{_OUT}/PLIP_Mask_Calculation/"    # Stage 1   → JSON files
RANDOM_MASK_OUTDIR = f"{_OUT}/Random_Mask_Calculation/"  # Stage 1.5 → JSON files
DEBUG_DIR          = f"{_OUT}/masked_smiles_lists/"       # per-ligand debug txt
PRED_DIR           = f"{_OUT}/predictions_txt/"           # Stage 2 raw SMILES txt
PLOT_DIR           = f"{_OUT}/plots/"                     # Stage 2 incremental PNGs
TEST_DIR           = f"{_OUT}/test/"                      # Stage 2 --test (wiped each run)
STAGE25_PRED_DIR   = f"{_OUT}/predictions_txt_random_pick/"  # Stage 2.5 random-pick predict
STAGE25_PLOT_DIR   = f"{_OUT}/plots_random_pick/"             # Stage 2.5 plots
STAGE27_PRED_DIR   = f"{_OUT}/predictions_txt_multi_seed/"    # Stage 2.7 multi-seed aggregated predictions
STAGE27_PLOT_DIR   = f"{_OUT}/plots_multi_seed/"              # Stage 2.7 multi-seed plots
STAGE3_DIR         = f"{_OUT}/stage3b_analysis/"           # Stage 3 grids + histograms
STAGE3_DIR_2_5        = f"{_OUT}/stage3.2.5_analysis/"     
STAGE3_DIR_2_7        = f"{_OUT}/stage3.2.7_analysis/"           # Stage 3 grids + histograms
STAGE4_DIR         = f"{_OUT}/stage4_br4_matching/"       # Stage 4 BR4 nearest-neighbour analysis
STAGE5_DIR        = f"{_OUT}/stage5_chembl_matching/"
CHEMBL_CACHE_PATH = f"{_OUT}/stage5_chembl_matching/chembl_brd4_cache.csv"
STAGE6_DIR         = f"{_OUT}/stage6_docking/"

#STAGE6_DIR         = f"/content/drive/MyDrive/GenAI4Drug/Mahzabeen/Output/expo02_docking/stage6_docking/"
STAGE7_DIR = f"{_OUT}/stage7_top_docked/"


# ── GNINA docking binary ───────────────────────────────────────────────────────
# Stage 6 will auto-download if the binary is not found at this path.
GNINA_BINARY       = GNINA_BINARY = f"{BASE_DIR}/gnina"
GNINA_DOWNLOAD_URL = "https://github.com/gnina/gnina/releases/download/v1.0.3/gnina"
# ── Stage 1 ligands ────────────────────────────────────────────────────────────
PIPELINE_INPUTS = [
    {"pdb_path": BASE_PDB_PATH + "4QZS", "plip_xml_path": BASE_XML_PATH + "4QZS",
     "resname": "JQ1", "chain": "A", "resseq": 201},
    {"pdb_path": BASE_PDB_PATH + "3MXF", "plip_xml_path": BASE_XML_PATH + "3MXF",
     "resname": "JQ1", "chain": "A", "resseq": 1},
    {"pdb_path": BASE_PDB_PATH + "3ZYU", "plip_xml_path": BASE_XML_PATH + "3ZYU",
     "resname": "1GH", "chain": "A", "resseq": 1173},
    {"pdb_path": BASE_PDB_PATH + "3P5O", "plip_xml_path": BASE_XML_PATH + "3P5O",
     "resname": "EAM", "chain": "A", "resseq": 1},
    {"pdb_path": BASE_PDB_PATH + "5HLS", "plip_xml_path": BASE_XML_PATH + "5HLS",
     "resname": "62G", "chain": "A", "resseq": 201},
]

INCLUDE_TYPES = [
    "hydrophobic", "hbond", "waterBridge",
    "saltBridge", "piStacking", "piCation", "halogen", "metal",
]

# ── ChemBERTa model ────────────────────────────────────────────────────────────
CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"

# ── Stage 1.5: random masking knobs ───────────────────────────────────────────
RANDOM_MASK_SEED = 17
# ── Stage 2.7: list of random seeds (each generates a full run, results aggregated) ──
RANDOM_MASK_SEEDS_LIST = [17, 53, 89]

# ────────────────────────────────────────────────────────────────────────────
MAX_GRID_MOLS = 99  # Assumption A3
# ────────────────────────────────────────────────────────────────────────────

# ── Stage 2: incremental generation knobs ─────────────────────────────────────
# ChemBERTa samples per (ligand × mask_count × strategy) cell.
# Total forward passes = sum_over_ligands(N_i × 2) × INCREMENTAL_NUM_SAMPLES
# Recommended range: 100 (fast/test) – 500 (publication quality).
INCREMENTAL_NUM_SAMPLES = 500
TOP_K                   = 20
TEMPERATURE             = 1.5

# ── Stage 2 (legacy full-mask knobs, kept for backward compatibility) ──────────
N_RANDOM_MASKED       = 20
MAX_MASKS_PER_INPUT   = 20
NUM_SAMPLES_PER_INPUT = 10
FULL_NUM_SAMPLES      = max(N_RANDOM_MASKED * NUM_SAMPLES_PER_INPUT, 2000)
GROUP_SEP             = " | "
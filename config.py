# -*- coding: utf-8 -*-
"""
py  ─  Shareduration for all pipeline stages.

▶ Edit BASE_DIR and USER_PREFIX, then all output paths update automatically.
"""

# Windows' console defaults to the cp1252 codepage, which can't encode the
# emoji used in this pipeline's print() calls (🚀 ✅ ⚠️ ...) and crashes with
# UnicodeEncodeError. Force UTF-8 on stdout/stderr here — the one module
# every stage script imports — so every stage is fixed at once. No-op on
# Colab/Linux, where stdout is already UTF-8.
import sys as _sys
if _sys.platform == "win32":
    for _stream in (_sys.stdout, _sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            try:
                _reconfigure(encoding="utf-8")
            except Exception:
                pass

# ── Root directory ─────────────────────────────────────────────────────────────
BASE_DIR    = "/group/bioinf/Users/Ishrak/output2"  # root of the pipeline tree (all outputs live under this)
USER_PREFIX = "Ishrak"
EXPERIMENT_TAG = "expo7"

# ── All outputs live under BASE_DIR / USER_PREFIX / Output / <subdir> ────────── old
#_OUT = f"{BASE_DIR}/{USER_PREFIX}/Output"

# ── All outputs live under BASE_DIR / USER_PREFIX / Output / <subdir> ──────────New way
_OUT = f"{BASE_DIR}/{USER_PREFIX}/Output/{EXPERIMENT_TAG}"

# ── Stage 1 inputs (not user-specific; shared raw data) ───────────────────────
BASE_PDB_PATH = f"{BASE_DIR}/Dummy_data/PDB/"
BASE_XML_PATH = f"{BASE_DIR}/Dummy_data/plip/"
REF_CSV_PATH  = f"{BASE_DIR}/Dummy_data/BR4_PDB_Data.csv"

# ── Output directories ─────────────────────────────────────────────────────────
STAGE0A_DIR        = f"{_OUT}/stage0a_chembl_download/"  # Stage 0a  → ChEMBL download + verified SMILES
MASK_CALC_OUTDIR   = f"{_OUT}/PLIP_Mask_Calculation/"    # Stage 1   → JSON files
RANDOM_MASK_OUTDIR = f"{_OUT}/Random_Mask_Calculation/"  # Stage 1.5 → JSON files
CHEMBL_MASK_OUTDIR = f"{_OUT}/ChEMBL_Mask_Calculation/"  # Stage 1.7 → ChEMBL random-mask JSONs
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
CHEMBL_PCHEMBL_MIN = 5.0   # used when Stage 1.7 must fetch ChEMBL (cache missing)
# CHEMBL_MASK_FRACTION = 0.25  # optional: fraction of heavy atoms to mask per ChEMBL mol;
#                               # if unset, stage1_7 prompts or uses PDB-derived rate (7D)
STAGE6_DIR         = f"{_OUT}/stage6_docking/"

#STAGE6_DIR         = f"/content/drive/MyDrive/GenAI4Drug/Mahzabeen/Output/expo02_docking/stage6_docking/"
STAGE7_DIR = f"{_OUT}/stage7_top_docked/"
STAGE8_DIR         = f"{_OUT}/stage8_analysis/"               # Stage 8 scatter + Pareto top-10
STAGE8_INPUT_CSV   = f"{_OUT}/stage6_docking/stage2/both/docking_summary.csv"  # override at prompt if absent
RDKIT_POLICY_LORA_DIR = f"{_OUT}/stage2_policy_lora/"         # Stage 2 RDKit-policy LoRA adapter


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
# Option A: smaller, trained on 100k ZINC — original baseline
CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
# Option B: BPE on 10M PubChem SMILES, richer vocabulary, same <mask> token — recommended
#CHEMBERTA_MODEL = "seyonec/PubChem10M_SMILES_BPE_450k"
# SELFIES ChemBERTa (BPE on SELFIES) — used by bpe_mask_adapter for Stage 1 SELFIES paths
CHEMBERTA_SELFIES_MODEL = "seyonec/BPE_SELFIES_PubChem_shard00_166_5k"
BPE_MASK_ADAPTER_ENABLED  = False  # adapter over-masks (cascades on atom-mapped SMILES); use 1-atom→1-<mask>
# Clean masking: mask exactly the requested atoms, keep all other atoms in
# ChemBERTa's native bare form (C, c, O) instead of bracketed [CH3], [cH].
# Only applies when BPE_MASK_ADAPTER_ENABLED is False.
CLEAN_SMILES_MASKING      = True
# Token-level masking: tokenize the CLEAN SMILES first, then mask whole BPE
# tokens that overlap the requested atoms (one <mask> per masked token).
# Precedence when adapter is off: TOKEN_LEVEL_MASKING > CLEAN_SMILES_MASKING.
# Set this True (and leave the adapter False) to use "tokenize-then-mask".
TOKEN_LEVEL_MASKING       = True
USE_STORED_MASKED_SMILES  = True   # Stage 1.9: read masked_smiles from JSON when indices match

# ── Stage 1.5: random masking knobs ───────────────────────────────────────────
RANDOM_MASK_SEED = 17
# ── Stage 2.7: list of random seeds (each generates a full run, results aggregated) ──
RANDOM_MASK_SEEDS_LIST = [17, 53, 89]

# ── Stage 1a: ChEMBL token-level random masking + single-shot generation ─────
STAGE1A_DIR           = f"{_OUT}/stage1a_random_token_masking/"  # one CSV per (percent, temperature, seed)
STAGE1A_MASK_PERCENTS = [5, 10, 15, 20, 25]   # % of BPE tokens masked per SMILES (direct token-level masking)
STAGE1A_TEMPERATURES  = [0.5,0.8,1.0, 1.2,1.5]   # ChemBERTa sampling temperatures
# Cap on how many Stage-0a SMILES are processed per run — a random sample of
# this size is drawn (reservoir sampling, one pass, seeded below) from the
# full chembl_verified_smiles.csv so the subset is representative of the
# whole file rather than just its first N rows. Set to None to process the
# entire file (no sampling).
STAGE1A_INPUT_LIMIT       = 500
STAGE1A_INPUT_SAMPLE_SEED = 42   # seed for the random sample above (reproducible across runs)

# ── Stage 1b large-scale PLIP mask calculation ────────────────────────────────
# Local PDB mirror: <PLIP_LARGE_SCALE_PDB_ROOT>/<mid2>/pdb<id>.ent.gz
# (mid2 = chars[1:3] of the 4-char id, e.g. "100d" -> "00" -> .../00/pdb100d.ent.gz)
PLIP_LARGE_SCALE_PDB_ROOT  = "/group/bioinf_tmp/Data/pdb"
# Pre-computed PLIP XML reports, flat: <PLIP_LARGE_SCALE_XML_ROOT>/pdb<id>.xml
PLIP_LARGE_SCALE_XML_ROOT  = "/group/bioinf_tmp/plip_pdb2xml"
STAGE1B_PLIP_MASK_DIR      = f"{_OUT}/stage1b_large_scale_plip_mask/"
STAGE1B_PLIP_SAMPLE_N      = 10000  # default sample size (overridable via --n)
STAGE1B_PLIP_SAMPLE_SEED   = 42    # default seed (overridable via --seed)
# "n"   : sample STAGE1B_PLIP_SAMPLE_N PDB/XML pairs at random (default)
# "all" : process every PDB/XML pair available (skip ones missing PDB or PLIP XML)
STAGE1B_SAMPLING_MODE      = "all"   # overridable via --sampling-mode {n,all}
STAGE1B_RESUME             = True  # overridable via --no-resume; skip pdb_ids already in the summary CSV

# ── Stage 1 / 1a: 2D interaction plot output (run_pipeline, shared) ──────────
# Applies to every run_pipeline() call — stage1_mask_calculation.py's 5-ligand
# main() as well as stage1a's large-scale batch — since the plot is generated
# inside the shared run_pipeline() function whenever out_prefix is set.
MASK_CALC_SAVE_PLOTS   = False     # False = skip 2D interaction plot generation entirely (saves disk + time)
MASK_CALC_PLOT_FORMAT  = "png"    # "png" (lossless, larger) or "jpg" (lossy, ~5-10x smaller)
MASK_CALC_PLOT_QUALITY = 85       # JPEG quality 1-95; only used when MASK_CALC_PLOT_FORMAT == "jpg"
MASK_CALC_PLOT_DIR     = f"{_OUT}/mask_calculation_plots/"  # separate tree from the .meta.json output dir

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

# ── Mask-decoding strategy (A/B switch) ───────────────────────────────────────
# False (default): generate_smiles_sequential — fill <mask>s one at a time,
#   re-running the model after each fill so every mask conditions on the tokens
#   already chosen (N forward passes for N masks). Captures inter-mask
#   dependencies → higher SMILES validity, slower.
# True: generate_smiles_oneshot — ChemBERTa's native MLM mode. ONE forward pass
#   predicts all <mask> positions simultaneously; each mask is sampled
#   independently from that single pass (conditionally independent, no
#   cross-mask awareness). Much faster, but usually lower validity/uniqueness as
#   the number of masks grows. Use to A/B against the sequential decoder.
ONESHOT_MASK_DECODING   = False

# When True, Stage 2.5 / 2.7 print per-mask_count sampling diagnostics to stdout:
#   • sample_draws     — candidate completions attempted (should == num_samples
#                        per freshly generated strategy cell)
#   • rdkit_validate   — Chem.MolFromSmiles calls on finished candidates
#                        (should == sample_draws per cell; no partial-string
#                        RDKit checks — those use a lightweight regex pass)
GENERATION_COUNT_DEBUG  = False

# ── Stage 2 (legacy full-mask knobs, kept for backward compatibility) ──────────
N_RANDOM_MASKED       = 20
MAX_MASKS_PER_INPUT   = 20
NUM_SAMPLES_PER_INPUT = 10
FULL_NUM_SAMPLES      = max(N_RANDOM_MASKED * NUM_SAMPLES_PER_INPUT, 2000)
GROUP_SEP             = " | "
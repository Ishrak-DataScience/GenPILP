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
BASE_DIR    = "/content/drive/MyDrive/GenAI4Drug/output"  # root of the pipeline tree (all outputs live under this)
USER_PREFIX = "Ishrak"
EXPERIMENT_TAG = "expo_finetuning_compound_loss"

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
STAGE9_LORA_DIR       = f"{_OUT}/stage9_property_lora/"       # Stage 9 masked-data property-guided LoRA adapter
STAGE9A_DIR           = f"{_OUT}/stage9a_property_eval_no_finetune/"  # Stage 9a baseline (no-fine-tune) property plots

# ── Stage 9: training-data source + re-masking rate ───────────────────────────
# Which masking data source(s) Stage 9 trains on:
#   "stage1a" : Stage 1a's ChEMBL random-token-masking output (STAGE1A_DIR) only
#   "stage1b" : Stage 1b's PLIP interaction-masking output (STAGE1B_PLIP_MASK_DIR) only
#   "both"    : concatenate both (default)
STAGE9_DATA_SOURCE = "both"   # "stage1a" | "stage1b" | "both"

# Percent of a molecule's ChemBERTa BPE tokens to mask, applied uniformly so
# every training pair Stage 9 sees is masked at the same rate regardless of
# source:
#   • Stage 1a: selects the pre-computed mask{STAGE9_MASK_PERCENT}pct_*.csv
#     combo file(s) instead of mixing every percent in STAGE1A_MASK_PERCENTS
#     together.
#   • Stage 1b: floor(STAGE9_MASK_PERCENT/100 * total_bpe_tokens) atom indices
#     are randomly re-sampled from the PLIP-derived candidate pool stored in
#     masked_atom_indices (the full non-interacting-atom set on
#     "non-attractive" rows, or the full interacting-atom set on "attractive"
#     rows) -- NOT the row's pre-built masked_smiles, which masks every atom
#     in that pool. If the pool itself has fewer atoms than the floor(N%)
#     target, every atom in the pool is masked (can't sample more than what
#     PLIP flagged).
STAGE9_MASK_PERCENT = 15
STAGE9_MASK_SEED     = 42   # seed for the deterministic per-molecule re-sampling above

# Override where Stage 9 (and Stage 9a) read their Stage-1a / Stage-1b
# training data from, for when masks were computed on a different machine
# than the one Stage 9 trains on (e.g. Stage 1b run on a cluster, its output
# CSV copied to a local path -- or a Drive tar.gz -- for training). Leave ""
# to read directly from STAGE1A_DIR / STAGE1B_PLIP_MASK_DIR (same machine,
# default). Only the dir/archive matching STAGE9_DATA_SOURCE needs to be
# set/present.
#
# May point at a plain directory OR a .tar/.tar.gz/.tgz/.tar.bz2/.tar.xz
# archive -- collect_pairs_from_stage1a/1b (stage9_masked_property_finetune.py)
# stream only the member(s) matching the expected filename pattern straight
# out of the archive (stage1b_large_scale_plip_mask_summary.csv for Stage 1b,
# mask{STAGE9_MASK_PERCENT}pct_temp*_seed*.csv for Stage 1a) without
# extracting the rest of the archive to disk.
#
# Either source alone is enough: a source whose location is missing/empty is
# warned about and skipped, not a hard stop (Stage 9a plots only the source(s)
# that loaded). The tar suffix is matched leniently too -- if the exact path
# below doesn't exist, a sibling file with the same stem and a different tar
# suffix (.tar / .tar.gz / _tar.gz ...) is used instead, with a warning.
# Points at the random-masking CONTROL ARM built by
# stage1a_random_masking_from_stage1b.py (see STAGE1A_FROM_STAGE1B_DIR below)
# so Stage 9a's two panels cover the SAME molecules. Set to "" to fall back to
# ordinary Stage 1a (ChEMBL molecules) in STAGE1A_DIR instead.
STAGE9_9A_STAGE1A_DATA_DIR = f"{_OUT}/stage1a_random_masking_from_stage1b/"
# Colab: Google Drive mounted at /content/drive, summary CSV read straight
# out of the tar.gz without full extraction (see note above).
STAGE9_9A_STAGE1B_DATA_DIR = "/content/drive/MyDrive/GenAI4Drug/Dummy_data/stage1b_output.tar.gz"

# ── Stage 9 / 9a: how many (masked, original) pairs to actually use ──────────
# Same convention as STAGE1A_INPUT_LIMIT above: instead of using every row the
# Stage 1a/1b output contains, a RANDOM SAMPLE of this size is drawn with a
# fixed seed, so a run is reproducible and a Colab session can be cut to a
# manageable size. None = use everything.
#
#   STAGE9_MAX_TRAINING_PAIRS
#       Cap for the REINFORCE training loop only. Applied to the COMBINED
#       Stage 1a + Stage 1b pool (that loop trains on one flat list).
#
#   STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE
#       Cap for the property-report pass -- Stage 9a's no-fine-tuning baseline
#       AND Stage 9's post-training plot. Applied PER SOURCE, not to the total,
#       so Stage 1a and Stage 1b panels get a comparable n instead of the
#       larger source swamping the sample. Sampling happens BEFORE generation,
#       so runtime scales with this number.
#
# Both scripts sample from a deterministic pair list with the same seed, so
# Stage 9a and Stage 9 still score exactly the same molecules -- the "before"
# and "after" figures stay directly comparable at any limit.
STAGE9_MAX_TRAINING_PAIRS        = None   # e.g. 5000
#       COMPARABILITY RULE: this number is part of what a property figure
#       MEANS. Stage 9a's baseline and Stage 9's post-training figure must be
#       produced at the SAME value, or the "before" and "after" panels are not
#       over the same molecules. The sample is seeded, so a capped run scores a
#       deterministic SUBSET of an uncapped one -- a capped Stage 9 figure is
#       still not comparable to an uncapped Stage 9a figure.
#
#       Left at None (uncapped) so a Stage 9 run lines up with an EXISTING
#       uncapped Stage 9a baseline. Once that comparison is made, set this to
#       e.g. 5000 for every later run -- eval is ~335k generations uncapped
#       against ~10k at 5000/source, and Wilson CIs are already tight at that n
#       (a rate near 0.5 is +/-1.4%). Re-run Stage 9a at the new value first.
#
#       Either script also takes "--limit N" (or "--limit none") to override
#       this for one run without editing config -- see their _parse_args.
#       Whichever mode produced a figure is recorded in its own footer:
#       "randomly sampled N of M usable pair(s), seed S" appears only on a
#       capped figure, so the two modes can always be told apart after the fact.
STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE = None    # None = score every available pair
STAGE9_PAIR_SAMPLE_SEED          = 42     # seed for both samples above

# ── Stage 9: cap TRAINING pairs per parent molecule ──────────────────────────
# The training-loop analogue of STAGE9_EVAL_DEDUP_BY_PARENT below, and the
# single most important knob in this block. Stage 1b writes one row per ligand
# INSTANCE, so a molecule resolved in N PDB entries contributes N training
# pairs -- and PDB instance counts follow crystallography, not chemistry.
# Measured on the shipped stage1b summary at STAGE9_MASK_PERCENT=15
# (334,550 surviving pairs over 44,977 distinct molecules, 7.4x redundancy):
#
#     33,538 pairs  [S@@](O)(=O)(=O)O                sulfate ion
#     20,249 pairs  C(O)[C@@H](O)CO                  glycerol (cryoprotectant)
#     18,876 pairs  C(O)[C@H](O)CO                   glycerol, other stereo-notation
#     14,549 pairs  C1[C@H](NC(C)O)...O1             NAG (glycosylation)
#     13,816 pairs  [S@](O)(=O)(=O)O                 sulfate, other stereo-notation
#     10,376 pairs  C(=O)(O)C                        acetate
#
# Those six alone are 33% of the training set and the top 100 parents are
# 58.5% -- i.e. without a cap the REINFORCE gradient is dominated by buffer
# components and cryoprotectants rather than by drug-like ligands.
#
# The cap keeps at most this many instances per unique canonical parent, drawn
# with a deterministic per-parent seed so different PLIP pockets (and therefore
# different mask positions) are represented rather than whichever PDB IDs sort
# first. It is applied PER SOURCE, never to the combined pool, so a molecule
# present in both Stage 1a and Stage 1b keeps one entry from each -- the
# random-masking vs PLIP-masking contrast is the point of training on both.
#
#   3    (default) keeps genuine mask-position diversity from distinct binding
#        pockets while removing the popularity weighting. 334,550 -> 68,772 pairs.
#   1    strict one-pair-per-molecule; matches the eval pass exactly and makes
#        every molecule contribute equally. 334,550 -> 44,977 pairs.
#   None no cap (the previous behaviour).
STAGE9_MAX_PAIRS_PER_PARENT      = 3

# ── Stage 9: epochs ──────────────────────────────────────────────────────────
# With the per-parent cap above, 68,772 pairs at STAGE9 BATCH_SIZE=16 is ~4,300
# optimizer steps per epoch. A rank-8 LoRA saturates long before the ~209,000
# steps the old uncapped 10-epoch setting implied, so 3 epochs is the default.
STAGE9_NUM_EPOCHS                = 3

# ── Stage 9.1: batched GPU forward + parallel RDKit scoring ──────────────────
# Knobs for "stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py",
# a drop-in variant of Stage 9 that optimises the two costs Stage 9 leaves on
# the table. It trains the SAME objective on the SAME pairs -- only the
# execution strategy differs -- so its adapter is directly comparable.
#
# Its output goes to its own directory so the two variants never overwrite each
# other's checkpoints or curves.
STAGE9_1_LORA_DIR = f"{_OUT}/stage9_1_property_lora_batched/"

# 7a -- one padded forward for a whole batch instead of one per molecule.
#   "auto"  enable when a CUDA device is present (default). The win is largest
#           on GPU, where per-launch overhead dominates a batch-1 forward.
#   True    force on (also valid on CPU, just a smaller win).
#   False   force off -- falls back to Stage 9's per-molecule rollout, which is
#           the reference implementation for the equivalence self-test.
STAGE9_1_BATCHED_ROLLOUT = "auto"

# 7b -- score the batch's molecules across a process pool instead of serially.
#   "auto"  min(cpu_count - 1, 8) workers, or serial when that is <= 1 (default).
#   N       exactly N workers.
#   0 or 1  serial, no pool (correct choice on a 2-vCPU Colab box, where pool
#           overhead can exceed the gain -- measure before trusting "auto").
# Workers run RDKit only. The Tox21 term stays on the parent process as one
# batched GPU forward; see score_tox21_batch and compute_property_components'
# need_tox21, and never raise this expecting the Tox21 model to parallelise.
STAGE9_1_SCORING_WORKERS = "auto"

# ── Stage 9.1: how hard to drive the GPU ─────────────────────────────────────
# READ THIS BEFORE RAISING THE BATCH SIZE.
#
# GPU-RAM usage is a BAD target to maximise. ChemBERTa is 44M parameters (~180
# MB in fp32) and LoRA trains 147k of them: no honest configuration will ever
# fill a 15 GB T4, and a run that did would not be a better run. What actually
# matters is wall-clock time and whether the adapter learns.
#
# The real trade-off is OPTIMIZER STEPS. Steps per epoch = pairs / batch_size,
# so on the ~65k capped Stage 1b pool:
#     batch  16 -> 4,070 steps/epoch  (12,210 over 3 epochs)
#     batch  64 -> 1,018 steps/epoch  ( 3,054 over 3 epochs)
#     batch 256 ->   254 steps/epoch  (   763 over 3 epochs)
# REINFORCE is a high-variance estimator and needs steps. Quadrupling the batch
# without touching anything else quarters the learning, and the run finishes
# sooner having learned less. If you raise the batch, raise STAGE9_LEARNING_RATE
# roughly in proportion (or raise STAGE9_NUM_EPOCHS) and watch the reward curve.
#
# 64 is the default: a real throughput gain over 16 now that 7a makes batch size
# mean something, while keeping ~1,000 steps per epoch.
STAGE9_1_BATCH_SIZE = 64          # int, or "auto" for token-budget batching below

# "auto" batch mode: pack each batch up to this many PADDED tokens
# (molecules x longest-in-batch) instead of a fixed molecule count. Keeps GPU
# memory near-constant regardless of molecule size -- big batches of short
# SMILES, small batches of long ones -- which is the safe way to run near the
# memory ceiling. Ignored when STAGE9_1_BATCH_SIZE is an int.
STAGE9_1_MAX_BATCH_TOKENS    = 65536
STAGE9_1_MAX_BATCH_MOLECULES = 512     # hard cap in "auto" mode

# Sort each epoch's pairs by token length so a batch holds similar-length
# molecules. Without it a single 500-token SMILES pads the whole batch out to
# 500, wasting most of the matrix -- your pairs range from ~8 to ~500 tokens, so
# this is a large effect. Batch ORDER is still shuffled every epoch, so the
# gradient sequence stays stochastic; only within-batch length is correlated.
STAGE9_1_LENGTH_BUCKETING = True

# Mixed precision. "auto" picks bf16 on Ampere+ (A100/L4), fp16 on Turing (the
# T4 in a standard Colab GPU runtime), and disables itself on CPU. Halves
# activation memory and speeds up the matmuls. Log-softmax, the KL and the
# log-probabilities are always computed in fp32 regardless -- they are the
# numerically delicate part and are cheap next to the forward pass.
#   "auto" | "bf16" | "fp16" | False
STAGE9_1_AMP = "auto"

# TF32 matmuls on Ampere+ (ignored on T4). Roughly free speed for this workload.
STAGE9_1_TF32 = True

# Count DISTINCT PARENT MOLECULES, not ligand instances, in the eval pass.
# Stage 1b writes one row per ligand instance (pdb_id:resname:chain:resseq), so
# a ligand resolved in 40 PDB entries would otherwise be scored 40 times -- 40
# attempts on the same parent, inflating every denominator toward whatever
# crystallography solved most often and making the cap above a cap on instances
# instead of molecules. True collapses each source to one pair per unique
# canonical parent SMILES BEFORE the sample is drawn, keeping the first
# occurrence (deterministic, so Stage 9a and Stage 9 still score the same set).
# False scores every instance, as before.
STAGE9_EVAL_DEDUP_BY_PARENT      = True

# ── Stage 9: composite score weights (all six terms configurable here) ────────
# score = w_valid*valid + w_qed*QED + w_sa*(1-SA/10) + w_novelty*(1-similarity_to_original)
#       + w_tox_alert*(1-PAINS/Brenk_alert) + w_tox21*(1-Tox21_classifier_toxic_prob)
STAGE9_SCORE_W_VALID     = 0.20
STAGE9_SCORE_W_QED       = 0.15
STAGE9_SCORE_W_SA        = 0.10
STAGE9_SCORE_W_NOVELTY   = 0.15
STAGE9_SCORE_W_TOX_ALERT = 0.15   # PAINS/Brenk structural-alert filter (RDKit built-in, always available)
STAGE9_SCORE_W_TOX21     = 0.25   # Tox21-classifier term (needs STAGE9_TOX21_MODEL_DIR below; contributes 0 if unset)

# ── Stage 9: KL anchor to the pretrained distribution ────────────────────────
# The REINFORCE objective on its own rewards score and nothing else, so nothing
# stops the policy drifting arbitrarily far from ChemBERTa's pretrained
# chemistry to chase reward (reward hacking / mode collapse). We therefore add
# a KL-control penalty (Jaques et al. 2017; the same device RLHF fine-tuning
# uses) against the FROZEN pretrained model:
#
#     loss = -(score - baseline) * sum(log_prob)  +  beta * sum_over_masks KL
#
# where KL is the exact, analytic KL(policy || pretrained) over the full vocab
# at each masked position. The reference distribution costs no extra memory:
# it is this same model with the LoRA adapter switched off
# (peft's disable_adapter()), so "pretrained" is exact by construction.
#
# beta = 0 disables the anchor and restores the pre-anchor behaviour exactly.
# Tune it by watching the reported mean KL per masked position: a healthy run
# drifts to a small but non-zero KL (order 0.1-1 nat), while a KL that keeps
# climbing means beta is too low, and a KL pinned at ~0 with a flat score means
# it is too high.
STAGE9_KL_BETA = 0.05

# The base model is FROZEN, so its train-mode dropout (p=0.1) only injects
# noise -- and it injects it into the KL as well: a freshly initialised adapter
# is mathematically identical to the reference (LoRA B=0), yet measured against
# a deterministic reference a train-mode policy pass reports 0.1-0.3 nats per
# position of purely spurious "drift". True spends the beta budget on real
# adapter drift instead by disabling dropout in the frozen base only (LoRA's
# own dropout, STAGE9 LORA_DROPOUT, stays active). Only applies when
# STAGE9_KL_BETA > 0; False restores stock dropout behaviour.
STAGE9_KL_BASE_DROPOUT_OFF = True

# ── Stage 9: Tox21 toxicity classifier (second, independent toxicity term) ───
# HuggingFace-style directory (AutoModelForSequenceClassification.from_pretrained
# -loadable, num_labels=12, multi_label_classification) fine-tuned on Tox21.
# Point this at your trained checkpoint; the term fails safe to 0 contribution
# until this directory exists (same convention as the SA-Score optional import).
STAGE9_TOX21_MODEL_DIR = ""   # TODO: set to your trained Tox21 ChemBERTa checkpoint directory
# Canonical Tox21 task order — MUST match your checkpoint's classification-head
# output order (this is DeepChem/MoleculeNet's standard load_tox21 task order).
STAGE9_TOX21_ALL_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
# Subset actually used when aggregating into the score (default: all 12; trim
# this list to focus on e.g. just NR-* or SR-* tasks without touching code).
STAGE9_TOX21_SELECTED_TASKS = list(STAGE9_TOX21_ALL_TASKS)
STAGE9_TOX21_AGGREGATION    = "mean"   # "mean" or "max" across STAGE9_TOX21_SELECTED_TASKS


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

# ── Stage 1a (from Stage 1b): the RANDOM-MASKING CONTROL ARM ─────────────────
# stage1a_random_masking_from_stage1b.py masks the SAME molecules Stage 1b
# PLIP-masked, but picks the masked tokens UNIFORMLY AT RANDOM instead of from
# PLIP's interaction pool. That is the control the Stage 9a figure's two panels
# are supposed to compare: same molecules, same percent, same evaluator --
# only the CHOICE of which tokens get masked differs. (Ordinary Stage 1a masks
# ChEMBL molecules instead, so its panel differs in molecule set AND mask
# choice at once, which is not a clean comparison.)
#
# Parents come from Stage 1b's summary CSV `smiles` column, deduplicated to
# unique canonical SMILES and filtered to the RDKit-valid ones.
STAGE1A_FROM_STAGE1B_DIR = f"{_OUT}/stage1a_random_masking_from_stage1b/"
# Percent of BPE tokens to mask. Defaults to STAGE9_MASK_PERCENT because
# Stage 9a looks for mask{STAGE9_MASK_PERCENT}pct_temp*_seed*.csv -- set this
# to anything else and Stage 9a will not find the file it writes.
STAGE1A_FROM_STAGE1B_MASK_PERCENT = STAGE9_MASK_PERCENT
STAGE1A_FROM_STAGE1B_SEED = 42   # base seed; per-molecule mask seed is derived from it
# How floor(15% of N tokens) is rounded: "floor" matches Stage 1b's
# _remask_from_pool exactly (fairest head-to-head), "round" matches ordinary
# Stage 1a's mask_smiles_tokens. Both mask at least one token.
STAGE1A_FROM_STAGE1B_ROUNDING = "floor"   # "floor" | "round"
# Cap on unique parents to mask (None = all of them).
STAGE1A_FROM_STAGE1B_LIMIT = None

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

# ── Stage 1c: upload local PDB/PLIP pairs to Google Drive ────────────────────
# Same eligibility rule as Stage 1b: only pdb_ids present as BOTH
# <PLIP_LARGE_SCALE_PDB_ROOT>/<mid2>/pdb<id>.ent.gz and
# <PLIP_LARGE_SCALE_XML_ROOT>/pdb<id>.xml are uploaded (as a pair).
GDRIVE_UPLOAD_MODE       = "n"     # "n" = upload GDRIVE_UPLOAD_N random pairs; "all" = every eligible pair
                                    # overridable via --upload-mode {n,all}
GDRIVE_UPLOAD_N          = 50      # pairs to upload when GDRIVE_UPLOAD_MODE == "n"; overridable via --n
GDRIVE_UPLOAD_SEED       = 42      # random sample seed; overridable via --seed
GDRIVE_FOLDER_ID         = ""      # target Drive folder ID; "" = My Drive root; overridable via --folder-id
# OAuth (installed-app) credentials — Google Cloud Console -> APIs & Services ->
# Credentials -> "OAuth client ID" -> Desktop app -> download as JSON.
# Used only if GDRIVE_SERVICE_ACCOUNT_FILE is unset/missing.
GDRIVE_CREDENTIALS_FILE  = f"{BASE_DIR}/{USER_PREFIX}/gdrive_oauth_client.json"
GDRIVE_TOKEN_FILE        = f"{BASE_DIR}/{USER_PREFIX}/gdrive_token.json"  # cached after first auth
# Service-account credentials (Google Cloud Console -> IAM & Admin -> Service
# Accounts -> Keys -> Create key -> JSON). Preferred on a headless server: no
# browser needed. The target Drive folder (GDRIVE_FOLDER_ID) must be shared
# with the service account's client_email, and — since service accounts have
# no personal storage quota — that folder should live on a Shared Drive.
GDRIVE_SERVICE_ACCOUNT_FILE = ""   # e.g. f"{BASE_DIR}/{USER_PREFIX}/gdrive_service_account.json"
GDRIVE_UPLOAD_MANIFEST   = f"{_OUT}/stage1c_gdrive_upload/upload_manifest.csv"
GDRIVE_UPLOAD_RESUME     = True    # overridable via --no-resume; skip pdb_ids already in the manifest CSV

# ── Stage 1 / 1a: 2D interaction plot output (run_pipeline, shared) ──────────
# Applies to every run_pipeline() call — stage1_mask_calculation.py's 5-ligand
# main() as well as stage1a's large-scale batch — since the plot is generated
# inside the shared run_pipeline() function whenever out_prefix is set.
MASK_CALC_SAVE_PLOTS   = True     # False = skip 2D interaction plot generation entirely (saves disk + time)
MASK_CALC_PLOT_FORMAT  = "png"    # "png" (lossless, larger) or "jpg" (lossy, ~5-10x smaller)
MASK_CALC_PLOT_QUALITY = 85       # JPEG quality 1-95; only used when MASK_CALC_PLOT_FORMAT == "jpg"
MASK_CALC_PLOT_DIR     = f"{_OUT}/mask_calculation_vis/"  # separate tree from the .meta.json output dir

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
ONESHOT_MASK_DECODING   = True

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
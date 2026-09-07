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

# ── Which machine this file is for ─────────────────────────────────────
# config.py picks ONE config_*.py per machine at import time, and only
# considers files that declare CONFIG_PLATFORM. The name is a label for the
# logs; what actually does the choosing is whether BASE_DIR below exists on
# the machine doing the import. Set GENPLIP_CONFIG=colab to force this one.
# Meant for: Google Colab, Drive mounted at /content/drive.
CONFIG_PLATFORM = "colab"

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
# ── SHARED masking rate (Stage 9, Stage 9a, Stage 10) ────────────────────────
# ONE value, so every stage masks at the same rate and their results stay
# comparable. Changing it changes what all of them mean, so change it once here
# and re-run every stage you intend to compare.
MASK_PERCENT = 15

STAGE9_MASK_PERCENT = MASK_PERCENT
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
STAGE9_NUM_EPOCHS                = 6

# ── Stage 9.1: batched GPU forward + parallel RDKit scoring ──────────────────
# Knobs for "stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py",
# a drop-in variant of Stage 9 that optimises the two costs Stage 9 leaves on
# the table. It trains the SAME objective on the SAME pairs -- only the
# execution strategy differs -- so its adapter is directly comparable.
#
# Its output goes to its own directory so the two variants never overwrite each
# other's checkpoints or curves.
STAGE9_1_LORA_DIR = f"{_OUT}/stage9_1_property_lora_batched/"


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 9.1 -- WHICH SPEEDUPS YOU ALLOW
# ═══════════════════════════════════════════════════════════════════════════
# Every hardware-tuned choice Stage 9.1 makes is a knob below. Nothing is
# decided by the machine behind your back: the script consults hardware
# detection only where you have written "auto".
#
# HOW THESE COMBINE (highest priority first)
#   1. a command-line flag        --speed / --workers / --no-batch
#   2. an explicit knob here      any STAGE9_1_* below that is not None
#   3. STAGE9_1_SPEED             the preset, which fills in every None
#
# So the normal way to work is: pick a preset, then override the one or two
# knobs you care about and leave the rest at None.
#
#   python "stage9_1_...finetuning.py" --hardware
#
# prints the machine profile AND every knob's resolved value, without training.
# Run that first on any new box -- it is the only way to see what "auto"
# actually became there.

# ── The preset ───────────────────────────────────────────────────────────────
#   "off"   No speedups at all: per-molecule rollout, serial scoring, fp32,
#           TF32 explicitly off, no length bucketing. This is the PARITY
#           setting -- the closest Stage 9.1 can run to Stage 9, for when the
#           run exists to be compared rather than to finish quickly. Expect it
#           to be several times slower.
#   "safe"  Only speedups that cannot change WHICH molecules get sampled or
#           what the objective is: the RDKit process pool, plus CPU thread and
#           pool tuning. Sampling order, batch composition and arithmetic
#           precision stay exactly as in "off". (Honest caveat: a different CPU
#           thread count can reorder BLAS reductions, so a CPU-ONLY run may
#           differ in the last bits. On GPU nothing changes.)
#   "fast"  DEFAULT, and exactly what this stage did before these knobs
#           existed: batched rollout, pooled scoring, AMP, TF32, bucketing.
#           Distributionally identical to "off" -- same objective, same
#           estimator -- but not token-for-token identical, because batching
#           consumes the RNG stream in a different order. That is expected and
#           is discussed at length in the script's module docstring.
#   "auto"  Alias of "fast"; every knob left at "auto" is then sized by
#           hardware_autotune (SLURM allocation, compute capability, ...).
#
# NOTE: the preset deliberately does NOT touch STAGE9_1_BATCH_SIZE. Batch size
# is a LEARNING knob here, not a speed one -- it sets the optimizer-step count,
# and REINFORCE needs steps. A preset that quietly quartered the learning in
# exchange for wall-clock would be lying about what it does.
STAGE9_1_SPEED = "fast"

# ── Per-knob overrides ───────────────────────────────────────────────────────
# None on any of these means "take it from STAGE9_1_SPEED". Set a value to pin
# that one choice regardless of the preset.

# 7a -- one padded forward for a whole batch instead of one per molecule.
#   "auto"  enable when a CUDA device is present. The win is largest on GPU,
#           where per-launch overhead dominates a batch-1 forward.
#   True    force on (also valid on CPU, just a smaller win).
#   False   force off -- falls back to Stage 9's per-molecule rollout, which is
#           the reference implementation for the equivalence self-test.
STAGE9_1_BATCHED_ROLLOUT = None

# 7b -- score the batch's molecules across a process pool instead of serially.
#   "auto"  defer to hardware_autotune: the allocation minus one core for this
#           process, capped at 16; serial when that is <= 1.
#   N       exactly N workers.
#   0 or 1  serial, no pool (often correct on a 2-vCPU Colab box, where pool
#           overhead can exceed the gain -- measure before trusting "auto").
# Workers run RDKit only. The Tox21 term stays on the parent process as one
# batched GPU forward; see score_tox21_batch and compute_property_components'
# need_tox21, and never raise this expecting the Tox21 model to parallelise.
STAGE9_1_SCORING_WORKERS = None

# Mixed precision. "auto" picks bf16 on Ampere+ (A100/L4), fp16 on Turing (the
# T4 in a standard Colab GPU runtime), and disables itself on CPU. Halves
# activation memory and speeds up the matmuls. Log-softmax, the KL and the
# log-probabilities are always computed in fp32 regardless -- they are the
# numerically delicate part and are cheap next to the forward pass.
#   "auto" | "bf16" | "fp16" | False
STAGE9_1_AMP = None

# Loss scaling for AMP.
#   "auto"  on for fp16, off for bf16 -- which is the rule, not a preference:
#           fp16's exponent range flushes small gradients to zero without it,
#           and bf16 has fp32's range and does not need it.
#   True / False to force. STAGE9_1_GRAD_SCALER_INIT_SCALE pins the starting
#   scale (default None = torch's 65536); lower it if fp16 gradients overflow
#   for the first few hundred steps.
STAGE9_1_GRAD_SCALER = None
STAGE9_1_GRAD_SCALER_INIT_SCALE = None

# TF32 matmuls on Ampere+ (no TF32 path on T4/Turing, so ignored there).
# Roughly free speed for this workload, at reduced mantissa precision.
# Setting this False now genuinely disables TF32 -- it previously could not,
# because the hardware profile switched TF32 on before this flag was read.
STAGE9_1_TF32 = None

# torch.set_float32_matmul_precision. "auto" follows STAGE9_1_TF32 ("high" when
# TF32 is on, "highest" when it is off), which is the honest pairing.
#   "auto" | "highest" | "high" | "medium"
STAGE9_1_MATMUL_PRECISION = None

# Sort each epoch's pairs by token length so a batch holds similar-length
# molecules. Without it a single 500-token SMILES pads the whole batch out to
# 500, wasting most of the matrix -- your pairs range from ~8 to ~500 tokens, so
# this is a large effect. Batch ORDER is still shuffled every epoch, so the
# gradient sequence stays stochastic; only within-batch length is correlated.
# It does change which molecules share a batch, which is why "safe" leaves it
# off and "fast" turns it on.
STAGE9_1_LENGTH_BUCKETING = None

# ── CPU threading ────────────────────────────────────────────────────────────
# Intra-op threads for THIS process (the model, the tokenizer, the Tox21 batch).
#   "auto"  the full detected allocation.
#   N       exactly N.
#   0       leave torch's own default alone entirely.
# Remember this process and its scoring workers share one CPU allocation: on a
# small box the right answer is often FEWER threads here, not more, since the
# parent spends most of each step waiting on the GPU and on pool.map.
STAGE9_1_TORCH_THREADS = None

# Threads INSIDE each scoring worker. "auto" is 1, and 1 is almost always
# right: W workers x T threads on a W-core allocation is the oversubscription
# that makes a pooled run slower than a serial one. Raise it only if you have
# deliberately left cores idle.
STAGE9_1_WORKER_BLAS_THREADS = None

# ── Scoring-pool shape ───────────────────────────────────────────────────────
# Process start method for the pool.
#   "auto"      "spawn" -- and that default is load-bearing, not cautious.
#   "fork"      faster startup (no re-import of the module tree) and legitimate
#               on a CPU-only Linux run, but fork copies this process's CUDA
#               context into every child and corrupts it. The script refuses
#               fork on Windows and downgrades it to spawn once CUDA is live.
#   "forkserver"
#               a server interpreter is started clean (no CUDA context), imports
#               the module tree ONCE, and every worker forks from it. Safe with
#               a live CUDA context, and the fastest pool startup on Linux and
#               Colab -- worth setting there, where spawn re-imports torch,
#               transformers and peft in every worker.
STAGE9_1_POOL_START_METHOD = "forkserver"

# pool.map chunksize divisor: chunk = len(batch) // (workers * factor).
# 1 gives one chunk per worker -- least IPC, worst tail latency if a single
# molecule is slow. Higher values re-balance at the cost of more IPC.
STAGE9_1_POOL_CHUNK_FACTOR = None

# Recycle each worker after this many tasks. None keeps workers for the whole
# run, which is what you want: under "spawn" a restart re-imports the module
# tree and rebuilds the PAINS/Brenk catalogs (~85 ms). Set an int only if you
# are chasing a slow memory leak in a long run.
STAGE9_1_POOL_MAXTASKSPERCHILD = None

# ── Memory ───────────────────────────────────────────────────────────────────
# Truncation length for the rollout, in tokens. The padded cost is
# B x L x V, so halving this halves the logits tensor. Setting it below the
# longest pair in your data silently truncates molecules, so reach for it only
# when a batch genuinely will not fit. A tokenizer reporting a SHORTER window
# still wins -- this is a ceiling, never a raise.
STAGE9_1_MAX_SEQ_TOKENS = None

# Rows per Tox21 forward pass; 0 means the whole batch in one forward (the
# default). The Tox21 classifier is the one score term that stays on the parent
# process, and at batch 512 that single [B, 256] forward through a second
# transformer can exhaust a small card well before the rollout does. Chunking
# is numerically exact -- the classifier is frozen and in eval(), so no row's
# logits depend on any other row.
STAGE9_1_TOX21_SUBBATCH = None

# ── Multi-GPU (only consulted under torchrun) ────────────────────────────────
# True treats STAGE9_1_BATCH_SIZE as the GLOBAL batch and splits it across
# ranks, so N GPUs run the SAME number of optimizer steps as one GPU and the
# speedup is real wall-clock. This is a correctness property, not an
# optimisation, which is why every preset holds it True.
# False makes the batch PER RANK: the effective batch grows N-fold and the run
# takes N times FEWER optimizer steps -- the trap described at length in the
# script's DDP section. Legitimate if that is what you want; never silent.
STAGE9_1_DDP_SPLIT_BATCH = None

# DistributedDataParallel's find_unused_parameters. False is correct here (LoRA
# adapters are fully used every step) and is faster -- True adds a graph
# traversal per backward. Set True only if DDP raises the "did not receive
# gradient" error after a model change.
STAGE9_1_DDP_FIND_UNUSED = None

# "auto" is nccl when GPUs are visible, gloo otherwise -- gloo being what lets
# the DDP path be exercised on a machine with no GPUs at all.
#   "auto" | "nccl" | "gloo"
STAGE9_1_DDP_BACKEND = None


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
#
# THIS KNOB IS NOT PART OF STAGE9_1_SPEED, on purpose -- see the note there.
#   int         fixed molecules per batch
#   "auto"      token-budget batching (see below)
#   "hardware"  sized by hardware_autotune from the dataset and the epoch
#               budget, under the STAGE9_1_HW_* limits further down
STAGE9_1_BATCH_SIZE = 64

# "auto" batch mode: pack each batch up to this many PADDED tokens
# (molecules x longest-in-batch) instead of a fixed molecule count. Keeps GPU
# memory near-constant regardless of molecule size -- big batches of short
# SMILES, small batches of long ones -- which is the safe way to run near the
# memory ceiling. Ignored when STAGE9_1_BATCH_SIZE is an int.
STAGE9_1_MAX_BATCH_TOKENS    = 65536
STAGE9_1_MAX_BATCH_MOLECULES = 512     # hard cap in "auto" mode

# "hardware" batch mode only. These were hardcoded inside hardware_autotune;
# they are here because they encode a judgement about YOUR run, not about the
# card. min_total_steps is the floor on optimizer updates the auto-sizer must
# leave you -- it is what stops a big GPU from choosing a batch so large that
# REINFORCE stops learning. mem_headroom is the fraction of the card the
# estimate is allowed to assume; the run prints its real peak after batch one,
# and that measurement beats this estimate every time.
STAGE9_1_HW_MIN_TOTAL_STEPS = 2000
STAGE9_1_HW_MAX_BATCH       = 512
STAGE9_1_HW_MEM_HEADROOM    = 0.55


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 10 -- supervised best-of-K fine-tuning (NO reinforcement learning)
# ═══════════════════════════════════════════════════════════════════════════
# Stage 9 optimises a composite SCORE with REINFORCE (higher = better).
# Stage 10 optimises a composite LOSS with ordinary backpropagation
# (lower = better), on the same pairs, at the same MASK_PERCENT, evaluated with
# the same property plots -- so the two are a controlled comparison of the
# TRAINING STRATEGY, not of the data or the metrics.
#
# How a non-differentiable RDKit loss reaches the weights
# -------------------------------------------------------
# It does not, and cannot, directly: Chem.MolFromSmiles is a C++ parser, so
# "loss = 1000 if invalid" is a Python float with no gradient. Stage 10 bridges
# it by SELECTION instead. For each masked molecule it samples K completions
# from one forward pass, scores each with the loss below, and trains ordinary
# cross-entropy toward the tokens of the best one. RDKit decides WHICH tokens
# to pull toward; the gradient itself is plain supervised CE.
#
# The per-candidate loss (all terms >= 0, lower is better)
# --------------------------------------------------------
#   valid molecule:  w_qed*(1 - QED)
#                  + w_sa*(SA_raw - 1)/9          linear over SA's 1..10 range
#                  + w_novelty*similarity(parent, generated)
#                  + w_tox_alert*PAINS_or_Brenk_alert
#   invalid:         S + w_valid,  where S = w_qed + w_sa + w_novelty + w_tox_alert
#
# S is the worst a VALID molecule can score, so an invalid one is always worse
# than every valid one by a margin of exactly w_valid. That is your "validity
# matters most" requirement, expressed in a bounded way: no 1000x term to
# destabilise the gradient, and w_valid is the single knob for how much worse
# invalid should be.
STAGE10_DIR = f"{_OUT}/stage10_supervised_bestofk/"

# "a" = CE toward the best candidate PLUS an unlikelihood term that actively
#       pushes DOWN the tokens of invalid candidates ("do not generate this").
# "b" = CE toward the best candidate only; bad candidates simply never become
#       targets. Simplest and most stable.
# Each variant writes to its own subdirectory, so both can be trained and
# compared without overwriting each other.
STAGE10_VARIANT = "a"          # "a" | "b"

STAGE10_NUM_CANDIDATES = 16    # K completions sampled per molecule per step
STAGE10_TOP_K          = 20    # candidate sampling; matches Stage 9 for comparability
STAGE10_TEMPERATURE    = 1.2   # candidate sampling; matches Stage 9 for comparability

# Loss weights. The first four define S (the worst a valid molecule can score);
# keeping them summed to 1.0 makes a valid molecule's loss land in [0, 1] and
# an invalid one at 1 + w_valid, which is easy to read off the curves.
STAGE10_W_QED       = 0.30
STAGE10_W_SA        = 0.20
STAGE10_W_NOVELTY   = 0.30
STAGE10_W_TOX_ALERT = 0.20
# Extra margin by which an invalid molecule beats the worst valid one.
# Raise it to weight validity harder; it never enters a valid molecule's loss.
STAGE10_W_VALID     = 1.00

# ── The parent fallback ──────────────────────────────────────────────────────
# MEASURED on this data at MASK_PERCENT=15: about 70% of molecules produce ZERO
# valid completions out of 16, and raising K from 8 to 16 does not change that
# (it is molecule-intrinsic, not sample luck). Those molecules have no good
# candidate to imitate, so the parent's own tokens are used as the target --
# always available and guaranteed valid.
#
# But the parent target teaches RECONSTRUCTION, and reconstructing the parent
# scores similarity = 1, i.e. the worst possible novelty. At full weight, ~70%
# of steps would train the model to copy, and novelty would collapse. This
# factor scales those steps only:
#     1.0 = full weight (fastest validity gain, highest risk of a copier)
#     0.3 = default, reconstruction acts as a background regulariser
#     0.0 = equivalent to skipping those molecules entirely
STAGE10_FALLBACK_WEIGHT = 0.3

# Variant "a" only: strength of the unlikelihood term -log(1 - P(token)) on
# invalid candidates. NOT scaled by STAGE10_FALLBACK_WEIGHT -- on a molecule
# with no valid candidate, pushing down 16 known-invalid completions is
# legitimate signal about what not to generate, unlike copying the parent.
STAGE10_UNLIKELIHOOD_WEIGHT = 0.5

# ── Which layers to train ────────────────────────────────────────────────────
# The LM head is ALWAYS trained (it is the "last layer" proper). This adds the
# last N encoder blocks on top. ChemBERTa-zinc-base-v1 has 6 blocks / 44.1M
# params in total:
#     0 -> lm_head only            0.6M params  (token preferences only)
#     1 -> + last block            7.7M params  <- RECOMMENDED
#     2 -> + last two blocks      14.8M params
# Note for the Stage 9 comparison: Stage 9's LoRA touches query/value in ALL
# blocks, so this is a different capacity budget, not just a different loss.
STAGE10_UNFREEZE_LAST_N_BLOCKS = 1

STAGE10_NUM_EPOCHS    = 10
STAGE10_BATCH_SIZE    = 16
STAGE10_LEARNING_RATE = 5e-5
STAGE10_GRAD_CLIP     = 1.0

# Reuse Stage 9's pair-collection knobs so both stages train on exactly the
# same molecules; set these to override for Stage 10 alone.
STAGE10_MAX_TRAINING_PAIRS   = None
STAGE10_MAX_PAIRS_PER_PARENT = None    # None = inherit STAGE9_MAX_PAIRS_PER_PARENT


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 10.1 -- Stage 10, made resumable and parallel-scored
# ═══════════════════════════════════════════════════════════════════════════
# Knobs for "stage10_1_parallel_RDKit_scoring_resumable_training.py".
#
# Stage 10.1 trains the SAME objective on the SAME pairs with the SAME batch
# size and the SAME optimizer-step count as Stage 10. It is an EXECUTION
# variant, not a scientific one, and it is bit-identical to Stage 10 given the
# same seed -- pinned by its self-test. Two things differ:
#
#   1. RESUMABILITY at optimizer-step granularity (Stage 10 resumes only at
#      epoch boundaries and rebuilds a fresh Adam, losing its moments).
#   2. RDKit candidate scoring runs across a process pool instead of serially.
#
# Why the pool is the win that matters here, unlike in Stage 9.1
# ---------------------------------------------------------------
# Stage 9 scores ONE molecule per optimizer step per sample. Stage 10 scores
# K = STAGE10_NUM_CANDIDATES (16) candidates for EVERY molecule, so RDKit is
# roughly 16x more dominant in Stage 10 than in Stage 9. At batch 16 that is
# 16 x 16 = 256 RDKit measurements per batch. At ~9 ms each (the figure Stage
# 9.1 measured) that is ~2.3 s of pure RDKit per batch, which on the ~65k
# capped Stage 1b pool works out to ~2.5 h per epoch and ~10 h for the four
# configured epochs. The pool is what removes that wall; the batched GPU
# forward that Stage 9.1 adds (its speedup 7a) is deliberately NOT ported,
# because it would change which candidates are sampled and so break the
# bit-identity above for a second-order gain.
STAGE10_1_DIR = f"{_OUT}/stage10_1_supervised_bestofk_parallel/"

# Worker processes for RDKit candidate scoring. Same contract as
# STAGE9_1_SCORING_WORKERS, and resolved by the same function:
#   "auto"  defer to hardware_autotune (SLURM allocation / cgroup quota / CPU
#           affinity, not os.cpu_count()), leaving one core for this process.
#   N       exactly N workers.
#   0 or 1  serial, no pool. A one-worker pool pays pickling and IPC for no
#           parallelism, so this is the right answer on a 2-vCPU Colab box.
STAGE10_1_SCORING_WORKERS = "auto"

# ── Checkpoint cadence ───────────────────────────────────────────────────────
# Write a full resumable checkpoint every N optimizer steps, on top of the
# per-epoch one. This is the knob that decides how much work a Ctrl-C, a
# pre-empted SLURM job or a dropped Colab runtime can cost you: at most N
# steps. Stage 10's effective value is "one epoch", i.e. hours.
#
# It is not free. Each write is the ~7.7M unfrozen parameters (~31 MB) plus
# Adam's two moment tensors for them (~62 MB), so ~93 MB per save. Writes are
# ATOMIC (temp file + os.replace) and ROLLING (one file, overwritten), so the
# disk cost is constant rather than cumulative and an interrupted write can
# never destroy the previous good checkpoint.
#   100  ~40 saves/epoch on the capped pool -- a good default on a local SSD.
#   500  fewer, larger gaps; sensible when save_dir is a mounted Google Drive,
#        where a 93 MB write is slow.
#   0    disable mid-epoch checkpointing (epoch boundaries only, like Stage 10).
STAGE10_1_CHECKPOINT_EVERY_STEPS = 0

# Keep the per-epoch epoch_NNN.pt snapshots alongside the rolling checkpoint.
# They are what lets you go back to "the model as of epoch 2" after the fact;
# the rolling checkpoint only ever holds the latest step. ~31 MB each.
STAGE10_1_KEEP_EPOCH_CHECKPOINTS = True

# Resume from an existing checkpoint WITHOUT asking. The prompt was only ever
# there for an interactive terminal -- a Colab `!python` cell, a nohup'd run or
# a SLURM batch job has no stdin and already auto-resumes -- and on a job you
# restart by hand the answer is always "yes". Resuming is also the
# non-destructive answer: "no" starts over and the first save overwrites the
# checkpoint you just declined.
#   True   never prompt; a checkpoint in save_dir is always continued.
#   False  prompt on a TTY, auto-resume when stdin is not one (the old
#          behaviour).
# Either way `--fresh` on the command line still wins and starts over.
# Stage 10.2 imports this prompt from Stage 10.1, so this knob governs it too.
STAGE10_1_AUTO_RESUME = True

# Batch size. None inherits STAGE10_BATCH_SIZE so the optimizer-step count --
# and therefore the learning dynamics -- are IDENTICAL to Stage 10 and the
# 10 vs 10.1 comparison measures wall-clock only. Setting an int here breaks
# that equivalence deliberately; if you raise it, read the note on optimizer
# steps under STAGE9_1_BATCH_SIZE first, and remember that supervised
# cross-entropy tolerates a larger batch far better than REINFORCE does.
STAGE10_1_BATCH_SIZE = None

# Seed for the epoch shuffle and the candidate sampling stream. 42 reproduces
# Stage 10's own permutation exactly (Stage 10 hardcodes random.Random(42 + 1)
# for a fresh run), which is what makes the two runs comparable batch for
# batch. Change it only to run a different replicate.
STAGE10_1_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 10.2 -- Stage 10, executed as hard as the machine allows
# ═══════════════════════════════════════════════════════════════════════════
# Knobs for "stage10_2_hardware_tuned_batched_AMP_DDP_training.py".
#
# Stage 10.1 took the ONE speedup that cannot change a number: RDKit scoring
# moved to a process pool, which runs the same pure function on the same
# strings in a different process. It deliberately stopped there, so that it
# stays bit-identical to Stage 10.
#
# Stage 10.2 takes the rest, and pays the price Stage 9.1 pays for them: a
# batched forward, mixed precision, TF32, length bucketing and multi-GPU. It is
# the same objective, the same loss weights, the same K and the same data.
# What it is NOT is bit-identical to Stage 10, and the reason is worth stating
# because it is specific to this stage rather than generic:
#
#     In Stage 9 the score MULTIPLIES a log-probability, so a perturbation of
#     the logits perturbs a gradient. In Stage 10 the score SELECTS the
#     training target. Sample the batch in one multinomial call instead of one
#     per molecule and the RNG stream is consumed in a different order; run the
#     logits through bf16 autocast and they move in the last few bits. Either
#     can make a DIFFERENT candidate win best-of-K, and then the model trains
#     toward a different token sequence -- not a numerically different update,
#     a different target.
#
# That is a real cost and it buys real wall-clock, so it is a choice rather
# than a default: STAGE10_2_SPEED = "off" reproduces Stage 10 exactly (the
# self-test asserts it), "safe" adds only the pool, "fast" takes everything.
# Compare "fast" against Stage 10 distributionally -- mean validity, mean loss,
# mean novelty over many molecules -- never by diffing outputs.
STAGE10_2_DIR = f"{_OUT}/stage10_2_supervised_bestofk_hardware_tuned/"

# ── The preset: which speedups this run is allowed ───────────────────────────
# Precedence, highest first:
#     command-line flag  >  config.STAGE10_2_<KNOB>  >  config.STAGE10_2_SPEED
# Any knob left None below follows the preset. Same contract, and the same
# resolver shape, as STAGE9_1_SPEED.
#
#   "off"   Stage 10's per-molecule forward and per-molecule sampler, serial
#           scoring, fp32, TF32 explicitly off, no bucketing. The parity
#           setting: bit-identical to Stage 10 and to Stage 10.1.
#   "safe"  Only what cannot change which candidates get sampled: the RDKit
#           pool, plus thread and pool tuning. Still bit-identical to Stage 10.
#           This is Stage 10.1's behaviour, reachable from this file.
#   "fast"  The default. Batched forward + batched sampling, pooled scoring,
#           AMP, TF32, length bucketing. Distributionally equivalent to Stage
#           10, NOT token-for-token identical -- see the note above.
#   "auto"  Alias of "fast"; every knob left "auto" is then sized by
#           hardware_autotune from the real allocation.
STAGE10_2_SPEED = "safe"

# ── Individually overridable (None = follow the preset) ──────────────────────
# One padded [B, L] forward per batch, and ONE multinomial over every masked
# position in the batch, instead of Stage 10's loop over molecules.
# "auto" = on when CUDA is present (batching mainly recovers per-launch
# overhead, which dominates a batch-1 forward on GPU and matters far less on
# CPU). This is the knob that breaks bit-identity with Stage 10; nothing else
# in this file changes which candidates are drawn.
STAGE10_2_BATCHED_FORWARD = None       # "auto" | True | False

# Worker processes for RDKit candidate scoring. Same contract as
# STAGE10_1_SCORING_WORKERS: "auto" defers to hardware_autotune (SLURM
# allocation / cgroup quota / CPU affinity, never os.cpu_count()), N is exact,
# 0 or 1 means serial because a one-worker pool pays IPC for no parallelism.
# This is the biggest single win in the whole family: at K=16 and batch 16,
# that is 256 RDKit measurements per optimizer step.
STAGE10_2_SCORING_WORKERS = "auto"     # "auto" | N | 0

# Autocast dtype for the forward pass. "auto" picks bf16 on Ampere+/Hopper
# (A100, L4, H100 -- no gradient scaler needed) and fp16 on Volta/Turing (T4,
# V100). False forces full fp32. The log-softmax, the cross-entropy and the
# unlikelihood term are ALWAYS evaluated in fp32 regardless: they are the
# numerically delicate parts and cost nothing beside the forward.
STAGE10_2_AMP = None                   # "auto" | "bf16" | "fp16" | False

# fp16 loss scaling. "auto" enables the scaler exactly when the resolved AMP
# dtype is fp16, and never for bf16, which keeps fp32's exponent range and so
# does not underflow. Not a speed knob -- it is forced by the dtype.
STAGE10_2_GRAD_SCALER = None           # "auto" | True | False
STAGE10_2_GRAD_SCALER_INIT_SCALE = None

# TF32 for the fp32 matmuls that remain, on Ampere and later. Setting False
# WRITES allow_tf32 = False rather than leaving torch's default, so the banner
# and the kernels cannot disagree.
STAGE10_2_TF32 = None                  # True | False
STAGE10_2_MATMUL_PRECISION = None      # "auto" | "highest" | "high" | "medium"

# Sort each epoch by token length before batching. Tokenised SMILES here run
# from ~8 to ~500 BPE tokens, so one long molecule in an unsorted batch pads
# every other row out to its length. Batch ORDER is reshuffled afterwards, so
# the gradient sequence stays stochastic rather than becoming a length
# curriculum. Note this changes batch COMPOSITION, so it breaks Stage 10
# parity even without AMP.
STAGE10_2_LENGTH_BUCKETING = None      # True | False

# Intra-op threads for THIS process, and BLAS threads inside each pool worker.
# 0 for the former means "leave torch's default alone", which is what "off"
# wants -- not touching a global is not the same as setting it to the value it
# already held. 1 for the latter is right whenever the pool is sized to the
# allocation: W workers x T threads on a W-core box is the oversubscription
# that makes a pooled run slower than a serial one.
STAGE10_2_TORCH_THREADS = None         # "auto" | N | 0
STAGE10_2_WORKER_BLAS_THREADS = None   # "auto" | N

# Pool shape. "spawn" is the default and is load-bearing, not cautious: fork
# copies this process's CUDA context into every child and corrupts it in ways
# that surface much later and elsewhere.
STAGE10_2_POOL_START_METHOD = None     # "auto" | "spawn" | "fork" | "forkserver"
STAGE10_2_POOL_CHUNK_FACTOR = None     # pool.map chunking divisor
STAGE10_2_POOL_MAXTASKSPERCHILD = None # recycle workers after N tasks | None

# Truncation ceiling for the padded forward. A memory knob more than a speed
# one: the padded cost is B x L x V, so halving L halves the logits tensor.
# Below the longest pair in the data this silently truncates molecules.
STAGE10_2_MAX_SEQ_TOKENS = None        # int

# ── Multi-GPU ────────────────────────────────────────────────────────────────
# Launch with `torchrun --nproc_per_node=N <script>`. The configured batch is
# treated as the GLOBAL batch and split across ranks, so the optimizer-step
# count matches a single-GPU run and the speedup is real wall-clock rather than
# fewer, larger updates. Setting this False is how you deliberately scale the
# effective batch with the world size; it never happens silently.
STAGE10_2_DDP_SPLIT_BATCH = None       # True | False
STAGE10_2_DDP_FIND_UNUSED = None       # True | False
STAGE10_2_DDP_BACKEND = None           # "auto" | "nccl" | "gloo"

# ── Sizing (deliberately NOT part of the preset) ─────────────────────────────
# Batch size is a LEARNING knob, not a speed knob: it sets the optimizer-step
# count. A preset called "fast" that quietly quartered the number of updates
# would be measuring something other than speed.
#   None        inherit STAGE10_BATCH_SIZE -- identical step count to Stage 10
#   int         exactly this many molecules per batch
#   "auto"      pack until molecules x longest-in-batch exceeds the token
#               budget below, so PADDED size (and therefore GPU memory) stays
#               roughly constant across short fragments and long peptides
#   "hardware"  sized by hardware_autotune from the dataset and epoch budget
#               (NOT from free GPU memory -- at 44M parameters the backbone
#               cannot fill a modern card, so memory is the wrong target)
STAGE10_2_BATCH_SIZE = None
STAGE10_2_MAX_BATCH_TOKENS = 65536     # "auto" mode: padded tokens per batch
STAGE10_2_MAX_BATCH_MOLECULES = 512    # "auto" mode: hard cap on molecules
STAGE10_2_HW_MIN_TOTAL_STEPS = 2000    # "hardware" mode: floor on updates
STAGE10_2_HW_MAX_BATCH = 512
STAGE10_2_HW_MEM_HEADROOM = 0.55

# ── Resumability (inherited wholesale from Stage 10.1) ───────────────────────
# Same meaning and same costs as STAGE10_1_CHECKPOINT_EVERY_STEPS: a full
# resumable checkpoint every N optimizer steps on top of the per-epoch one, so
# a Ctrl-C, a pre-empted SLURM job or a dropped Colab runtime costs at most N
# steps. ~93 MB per write (unfrozen params + Adam's two moment tensors),
# atomic and rolling, so the disk cost is constant. 0 disables mid-epoch saves.
STAGE10_2_CHECKPOINT_EVERY_STEPS = None   # None inherits STAGE10_1's value
STAGE10_2_KEEP_EPOCH_CHECKPOINTS = None   # None inherits STAGE10_1's value
STAGE10_2_SEED = None                     # None inherits STAGE10_1_SEED


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 10.3 (Tox21 classifier)  +  STAGE 10.4 (generator that uses it)
# ══════════════════════════════════════════════════════════════════════════════
# Two scripts, two knob prefixes, and the split is worth holding onto:
#
#   stage10_3_tox21_train.py           STAGE10_3_TOX21_*   trains the Tox21
#                                      classifier from Tox21/tox21.csv
#   stage10_4_tox21_aware_training.py  STAGE10_4_*         trains the GENERATOR,
#                                      reading that classifier as a loss term
#
# Stage 10.2 changed HOW Stage 10 runs and left the objective alone. Stage 10.4
# does the opposite: it reuses every STAGE10_2_* execution knob above, unchanged
# and by design, and adds one term to the loss.
#
#     loss(valid) = w_qed*(1-QED) + w_sa*(SA-1)/9 + w_novelty*similarity
#                 + w_tox_alert*alert + w_tox21*(1 - Tox21_clean_probability)
#
# There is deliberately NO STAGE10_4_SPEED. Holding execution identical to
# Stage 10.2 is what makes a 10.2-vs-10.4 comparison a measurement of the
# objective instead of a confound of two things changing at once.
STAGE10_4_DIR = f"{_OUT}/stage10_4_supervised_bestofk_tox21_aware/"

# ── the weight, and what it does to the loss scale ───────────────────────────
# ADDED ON TOP of Stage 10's four weights rather than renormalising them, so
# every existing term keeps its exact meaning and 10.3's qed / sa / novelty /
# tox_alert curves stay directly comparable with 10.2's. Only the totals move:
#
#     worst VALID molecule   1.00 -> 1.20   (= 0.30+0.20+0.30+0.20 + 0.20)
#     any INVALID molecule   2.00 -> 2.20   (= worst valid + w_valid 1.00)
#
# The margin between them is still exactly w_valid, which is the invariant that
# encodes "validity matters most"; stage10_3's self-test asserts it.
#
# 0.20 matches w_tox_alert: the two toxicity notions -- a curated substructure
# RULE and a fitted assay MEASUREMENT -- start out weighted equally, and the
# stage10.4a_toxicity_comparison.png figure is what tells you whether they
# actually agree on generated molecules. Set to 0 to run Stage 10.2's exact
# objective through Stage 10.4's code path.
STAGE10_4_W_TOX21 = 0.20

# ── STAGE 10.3: the classifier this term reads ───────────────────────────────
# Stage 10.4 REFUSES TO START with w_tox21 > 0 and no checkpoint, and that is
# not pedantry. score_tox21 fails safe to 0.0 ("maximally toxic") when
# unconfigured, so every candidate would be charged the identical full weight
# -- a constant, invisible to an argmin over K candidates. The run would train
# exactly like Stage 10.2 while plotting a toxicity curve pinned at zero, and
# nothing about it would look broken.
#
# Build one with:   python stage10_3_tox21_train.py
# then set STAGE9_TOX21_MODEL_DIR (further down, in the Stage 9 block) to the
# directory it writes. The knobs below configure that training run.
STAGE10_3_TOX21_CSV = "Tox21/tox21.csv"
STAGE10_3_TOX21_MODEL_OUT = f"{_OUT}/stage10_3_tox21_classifier/"
STAGE10_3_TOX21_BASE_MODEL = None   # None = CHEMBERTA_MODEL, the generator's
                                    # own backbone (same tokenizer, so the
                                    # classifier sees the same token stream the
                                    # generator produces)
STAGE10_3_TOX21_EPOCHS = 6
STAGE10_3_TOX21_BATCH_SIZE = 32
STAGE10_3_TOX21_LR = 2e-5
STAGE10_3_TOX21_WEIGHT_DECAY = 0.01
STAGE10_3_TOX21_GRAD_CLIP = 1.0
# MUST match stage9.score_tox21's truncation (hard-coded 256) or the classifier
# trains on whole molecules and scores truncated ones. The trainer warns if you
# change it.
STAGE10_3_TOX21_MAX_TOKENS = 256
# "scaffold" groups Bemis-Murcko cores so no test compound shares a core with a
# training one. It typically costs 0.05-0.10 AUC against "random" -- that gap is
# the measurement, not a regression, and the honest number for molecules Stage
# 10.3 generates (which are novel by construction).
STAGE10_3_TOX21_SPLIT = "scaffold"      # "scaffold" | "random"
STAGE10_3_TOX21_VAL_FRAC = 0.10
STAGE10_3_TOX21_TEST_FRAC = 0.10
STAGE10_3_TOX21_SEED = 42
# Tox21 positives run 2.9% (NR-PPAR-gamma) to 16.2% (SR-ARE). Without
# pos_weight the minimiser finds "predict inactive everywhere", which gives
# ~0.5 AUC and -- worse for Stage 10.3 -- a near-constant score_tox21.
STAGE10_3_TOX21_POS_WEIGHT = True
STAGE10_3_TOX21_POS_WEIGHT_CAP = 50.0   # one rare assay's gradient must not
                                        # drown the other eleven

# ── resumability ─────────────────────────────────────────────────────────────
# A full checkpoint (weights + AdamW moments + the LR scheduler + RNG streams +
# history) is written after every epoch, atomically and rolling, so a dropped
# Colab runtime or a pre-empted SLURM job costs at most one epoch. Re-running
# the same command continues; --fresh retrains from the base model.
#
# It is ~530 MB, not Stage 10's ~93 MB, because this is a FULL fine-tune: all
# ~44M parameters are trainable and Adam keeps two moments for each. Rolling,
# so that cost is constant rather than per-epoch.
STAGE10_3_TOX21_AUTO_RESUME = None   # None = ask on a TTY, auto-resume when
                                     # there is no terminal. True = never ask.
STAGE10_3_TOX21_KEEP_CHECKPOINT = True   # False deletes it once the run ends,
                                         # which also makes the run unextendable
# EXTENDING a finished run to more epochs cannot be exact. OneCycleLR's shape is
# a function of total_steps, so a 6-epoch cycle and a 10-epoch cycle are
# different curves, not a prefix and its continuation -- by epoch 6 the first
# has annealed to ~0 while the second is only 60% through its anneal.
#   "restart"  rebuild the cycle for the new length and fast-forward to the
#              current step. The LR jumps back up: a warm restart. Better for
#              escaping a mediocre minimum.
#   "hold"     keep the LR the previous cycle ended on, flat, for the added
#              epochs. Fewer surprises; less able to move far.
# Either way the run prints both LR values rather than changing them quietly.
STAGE10_3_TOX21_EXTEND_LR = "restart"    # "restart" | "hold"

# ── STAGE 10.4: how the tox21 forward is executed during generator training ──
# The classifier runs as ONE batched forward on the training process, never in
# the RDKit worker pool -- a torch model per worker would multiply its memory
# by the pool size and have every worker contend for the same GPU. This is
# Stage 9.1's split, and stage9_1_scoring_worker.py documents it.
#
# Stage 10 scores K candidates per molecule where Stage 9 scored one, so at
# B=16, K=16 that is 256 classifier rows per optimizer step. Two exact
# reductions, both on by default, both switchable so the self-test can check
# them against the unreduced path:
STAGE10_4_TOX21_VALID_ONLY = True   # invalid candidates take the loss's invalid
                                    # branch and never read the term, so
                                    # scoring them is a discarded forward pass
                                    # (~70% of candidates at MASK_PERCENT=15)
STAGE10_4_TOX21_DEDUP = True        # the classifier is frozen and in eval(), so
                                    # identical SMILES give identical scores;
                                    # K draws from a top-20 distribution repeat
STAGE10_4_TOX21_SUBBATCH = 0        # rows per classifier forward; 0 = the whole
                                    # batch at once. Raise off 0 only if that
                                    # single [N, 256] allocation is what OOMs.

# ── sizing and resumability (None = inherit Stage 10.2's value) ──────────────
STAGE10_4_BATCH_SIZE = None
STAGE10_4_CHECKPOINT_EVERY_STEPS = None
STAGE10_4_KEEP_EPOCH_CHECKPOINTS = None
STAGE10_4_SEED = None


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 10 FAMILY -- one output folder, or three?

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 10 FAMILY -- shared train/validation split
# ══════════════════════════════════════════════════════════════════════════════
# Read by stage10_data_split.py and used by ALL FOUR Stage 10 trainers, so the
# word "validation" means the same molecules in 10, 10.1, 10.2 and 10.4.
#
# Before this existed, every collected pair was trained on and the headline
# property figure was computed over that same pool. That matters more here than
# in most pipelines: the parent fallback makes reconstruction the modal training
# signal (~70% of molecules at MASK_PERCENT=15), so a model that merely
# memorised its parents would have scored well and looked correct.
#
# The partition is on BEMIS-MURCKO FRAMEWORKS, not on rows or molecules. Rows
# are hopeless (sulfate alone spans 33,538 of them); molecules still let close
# analogues straddle the split. Whole framework groups move together, so no
# validation molecule shares a ring system with a training one. Expect held-out
# numbers to sit BELOW random-split numbers -- that gap is the memorised
# chemistry, and removing it is the point.
# "scaffold" groups whole Murcko cores -- the only mode whose held-out number
# is a generalisation estimate. "random" shuffles molecules instead: it reads
# several points better BECAUSE analogues of training molecules are in the
# validation fold, so use it as a COMPARISON against scaffold, not a
# replacement. Build either with:  python stage10_data_split.py --random
STAGE10_SPLIT_MODE = "scaffold"   # "scaffold" | "molecule" | "random"
STAGE10_VAL_FRAC   = 0.10
STAGE10_TEST_FRAC  = 0.0          # 0 disables the third fold; the held-out
                                  # property report already uses the val fold
STAGE10_SPLIT_SEED = 42

# Computed once over EVERY parent and cached here, so the split does not depend
# on STAGE10_MAX_TRAINING_PAIRS, on STAGE9_MAX_PAIRS_PER_PARENT or on which
# variant is running -- a molecule keeps its fold across every run.
# The mode is appended to this name ("..._split.scaffold.json",
# "..._split.random.json"), so several splits coexist and switching
# STAGE10_SPLIT_MODE never silently rebuilds over the other one.
STAGE10_SPLIT_MANIFEST = f"{_OUT}/stage10_split.json"

# Validation runs the full best-of-K procedure (K RDKit calls per molecule), so
# it is capped. The same molecules are drawn every epoch and every run, or the
# curve would move for reasons that are not the model. 0 = the whole fold.
STAGE10_VAL_MAX_PAIRS = 2000

# True: the epoch with the lowest mean held-out loss becomes the final model,
# written the moment it improves so an interrupted run keeps the best epoch
# rather than the last. False: the last epoch wins, as before.
STAGE10_SELECT_BEST_VAL = True

# ═══════════════════════════════════════════════════════════════════════════
# Stage 10, 10.1 and 10.2 train the SAME objective on the SAME pairs and differ
# only in execution. This flag decides whether that shared objective also means
# a shared output directory and a shared, continuable checkpoint.
#
#   False (default)  Each stage writes to its own directory (STAGE10_DIR,
#                    STAGE10_1_DIR, STAGE10_2_DIR) and resumes only its own
#                    checkpoints. Three independent runs of one objective,
#                    which is what makes a 10 vs 10.1 vs 10.2 timing
#                    comparison a comparison of execution and nothing else.
#
#   True             All three write to STAGE10_SHARED_DIR/variant_<v>/ and
#                    share ONE rolling checkpoint, "stage10_lineage.pt". A run
#                    interrupted under any of them continues under any other,
#                    at the exact batch it stopped on, with Adam's moments and
#                    the RNG streams intact. Start on the reference
#                    implementation, move to the tuned one when a GPU frees
#                    up, finish wherever.
#
# WHAT SHARED MODE CHANGES, BEYOND THE PATH
# ------------------------------------------
#   * Stage 10 gains optimizer state. Its own format saves only the unfrozen
#     tensors, so today every Stage 10 resume silently rebuilds Adam from
#     scratch and discards exp_avg / exp_avg_sq. In shared mode it writes and
#     reads the lineage checkpoint instead, and stops doing that. (With the
#     flag off, Stage 10 keeps its existing epoch_NNN.pt + JSON path exactly,
#     so nothing about an existing run changes.)
#
#   * There is ONE final model at the shared root -- whichever stage finished
#     the lineage last, which is the correct reading if the three are
#     continuing a single run. Training curves and property figures keep their
#     per-stage filenames (stage10a_*, stage10.1a_*, stage10.2a_*), so no
#     diagnostic is ever overwritten.
#
#   * A lineage may MIX execution paths, and that changes what the run is.
#     Epochs trained under Stage 10.2's bf16 batched sampling are not epochs
#     trained under Stage 10's fp32 per-molecule sampling -- the objective is
#     identical but the targets selected are not the ones the other path would
#     have selected. Every save therefore appends to a provenance log naming
#     which stage trained which epochs under which precision, and a resume
#     across a change prints a banner saying so. The log is kept in plain text
#     in stage10_lineage.json next to the weights, because `model.safetensors`
#     has no memory of the dtype it was trained under and the terminal
#     scrollback will be gone by the time anyone asks.
#
# Variants never merge: 10a and 10b are different objectives (10a adds the
# unlikelihood term), so each keeps its own variant_<v> subdirectory in both
# modes. Only execution is allowed to vary along a lineage.
STAGE10_SHARED_OUTPUT = False
STAGE10_SHARED_DIR = f"{_OUT}/stage10_family_shared/"


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
STAGE9_SCORE_W_VALID     = 0.25
STAGE9_SCORE_W_QED       = 0.20
STAGE9_SCORE_W_SA        = 0.15
STAGE9_SCORE_W_NOVELTY   = 0.15
STAGE9_SCORE_W_TOX_ALERT = 0.25   # PAINS/Brenk structural-alert filter (RDKit built-in, always available)
STAGE9_SCORE_W_TOX21     = 0.0   # Tox21-classifier term (needs STAGE9_TOX21_MODEL_DIR below; contributes 0 if unset)

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
# Resume from (and APPEND to) an arbitrary existing summary CSV instead of the
# one inside STAGE1B_PLIP_MASK_DIR -- for topping up a summary carried over
# from an earlier run or another machine. Its pdb_ids are skipped and new rows
# are appended to that same file; its header must match the script's columns
# exactly or the run aborts. "" = use STAGE1B_PLIP_MASK_DIR's own summary CSV.
# Overridable via --resume-from. Meaningless together with --no-resume.
STAGE1B_RESUME_FROM_CSV    = ""
# A pdb_id counts as "done" if it appears in the summary CSV at all, INCLUDING
# as a status=="error" row, so failed structures are not retried by default.
# True re-queues them: every id owning at least one error row is dropped from
# the CSV (rewritten atomically, once, at startup) and reprocessed, so the
# retry replaces those rows rather than duplicating the id's successful ones.
# Overridable via --retry-errors.
STAGE1B_RETRY_ERRORS       = False
# Seed for the sampled verification-figure draw (see MASK_CALC_SAVE_PLOTS
# below). Changing it re-rolls WHICH binding sites get plotted without redoing
# any masking -- useful for spot-checking a fresh sample via --plot-only.
STAGE1B_PLOT_SEED          = 42

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
# inside the shared run_pipeline() function.
#
# Stage 1b reads MASK_CALC_SAVE_PLOTS as TRI-STATE, because drawing a
# 12x12in/300dpi figure for every one of its ~500k binding sites is by far the
# most expensive thing in that run and exists only to eyeball that the masking
# is right:
#     False / 0   no figures at all
#     True        a figure for every masked binding site
#     int N > 0   figures for exactly N randomly sampled binding sites
# The N sample is drawn from the finished summary CSV once masking completes
# (seeded by STAGE1B_PLOT_SEED), so the count is exact and the pass can be
# replayed on its own with --plot-only. Stage 1's own 5-ligand main() just
# treats any non-zero value as "on", so a number here is safe for it too.
# Overridable via --plot-sample-n.
MASK_CALC_SAVE_PLOTS   = 200      # 0/False = none, True = every site, N = N random sites (Stage 1b)
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
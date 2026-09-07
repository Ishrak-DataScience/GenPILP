# -*- coding: utf-8 -*-
"""
stage10_2_hardware_tuned_batched_AMP_DDP_training.py
=====================================================
Stage 10.2 -- Stage 10's supervised best-of-K fine-tuning, executed as hard as
the machine allows.

SAME loss, SAME data, SAME K, SAME optimizer-step count. Everything scientific
is imported from stage10_vanila_backpropagation_training and
stage10_1_parallel_RDKit_scoring_resumable_training rather than re-implemented,
so if the objective changes there it changes here.

Where this sits in the family
------------------------------
    Stage 10     the reference: one forward per molecule, one RDKit call per
                 candidate, fp32, resumable at epoch boundaries only.
    Stage 10.1   + RDKit scoring in a process pool, + resumable at step
                 granularity with Adam's moments and the RNG streams. It stops
                 there ON PURPOSE, and so stays BIT-IDENTICAL to Stage 10.
    Stage 10.2   + everything Stage 9.1 does: one padded [B, L] forward per
                 batch, one multinomial for every masked position in the
                 batch, bf16/fp16 autocast, TF32, length bucketing, thread and
                 pool sizing from the real allocation, and multi-GPU under
                 torchrun.

Why Stage 10.1 stopped, and why this file does not
---------------------------------------------------
Stage 9.1 accepts that batching changes the order in which the global RNG is
consumed, because there the score MULTIPLIES a log-probability: a different
sample is a different draw from the same distribution, and the estimator is
unbiased either way.

Stage 10 is not like that. Here the score SELECTS the training target. Draw the
whole batch in one multinomial instead of one call per molecule, or run the
logits through bf16 autocast, and a different candidate can win best-of-K --
at which point the model trains toward a DIFFERENT TOKEN SEQUENCE. The
difference stops being numerical and becomes a different target. Stage 10.1
judged that too high a price for a second-order gain and kept the per-molecule
path.

There is a third source of divergence, easy to miss and worth naming because it
survives every seed: DROPOUT DRAWS ONCE PER FORWARD PASS. Stage 10 runs B
forwards per batch and so consumes B dropout masks; the batched path runs one
and consumes one. Measured on this model, that alone moves the mask logits by
~1.2 -- larger than anything AMP does. It cannot be aligned by seeding, because
the two paths do not ask the RNG for the same shapes. So the batched path is
not "Stage 10 plus rounding error"; it is a legitimately different sample from
the same training procedure, and the self-test asserts the two agree only once
dropout is switched off and top_k is 1, where nothing random remains.

This file makes the opposite choice, deliberately, and makes it reversible:

    config.STAGE10_2_SPEED = "off"    per-molecule forward, per-molecule
                                      sampler, serial scoring, fp32, no
                                      bucketing. BIT-IDENTICAL to Stage 10 and
                                      to Stage 10.1 -- asserted by the
                                      self-test, not merely intended.
    config.STAGE10_2_SPEED = "safe"   + the RDKit pool only. Still
                                      bit-identical (a pure function of two
                                      strings, run in another process).
    config.STAGE10_2_SPEED = "fast"   the default: everything above. Compare
                                      it against Stage 10 DISTRIBUTIONALLY --
                                      mean validity, mean loss, mean novelty
                                      over many molecules -- never by diffing.

So the same file spans "prove it matches" and "make it finish", and which one
you get is written down in config rather than decided by the machine.

Where the time actually goes
-----------------------------
Stage 10 scores K = 16 candidates per molecule against Stage 9's one, so RDKit
dominates here far more than it dominates Stage 9. At batch 16 that is 256
measurements per optimizer step, ~2.3 s of pure RDKit, against roughly 16
forward passes that batching collapses into one. The ordering of the speedups
follows from that:

    7b  the scoring pool          the win. Removes most of the wall clock.
    7a  the batched forward       real but smaller, and it is the one that
                                  costs bit-identity.
    AMP / TF32                    shrinks the [B, L, V] logits and the matmuls
                                  behind them; matters at large batch.
    length bucketing              8-to-500-token SMILES, so an unsorted batch
                                  wastes most of its padded matrix.
    DDP                           linear in ranks, on the wall clock only --
                                  the batch is split so the step count holds.

Configuration -- you choose which speedups this run is allowed
---------------------------------------------------------------
Precedence, highest first:

    command-line flag  >  config.STAGE10_2_<KNOB>  >  config.STAGE10_2_SPEED

CONFIG IS AUTHORITATIVE OVER hardware_autotune, never the other way round. A
knob may say "auto", and only then is the machine profile consulted. Nothing
here writes a torch backend global that config did not ask for -- see
apply_hardware_settings for the specific defect that rule exists to prevent.

  Individually overridable (None = follow the preset):
      _BATCHED_FORWARD  one padded forward + one sampler call  "auto"|True|False
      _SCORING_WORKERS  RDKit process pool                     "auto"|N|0
      _AMP              autocast dtype           "auto"|"bf16"|"fp16"|False
      _GRAD_SCALER      fp16 loss scaling        "auto"|True|False (+_INIT_SCALE)
      _TF32             TF32 matmuls on Ampere+                 True|False
      _MATMUL_PRECISION "auto"|"highest"|"high"|"medium"
      _LENGTH_BUCKETING sort each epoch by token length         True|False
      _TORCH_THREADS    intra-op threads here          "auto"|N|0 (leave alone)
      _WORKER_BLAS_THREADS  threads inside each pool worker     "auto"|N
      _POOL_START_METHOD    "auto"|"spawn"|"fork"|"forkserver"
      _POOL_CHUNK_FACTOR / _POOL_MAXTASKSPERCHILD
      _MAX_SEQ_TOKENS   padded-forward truncation ceiling
      _DDP_SPLIT_BATCH / _DDP_FIND_UNUSED / _DDP_BACKEND

  Sizing (NOT part of the preset -- batch size sets the optimizer-step count):
      _BATCH_SIZE  None (inherit Stage 10) | int | "auto" | "hardware"
      _MAX_BATCH_TOKENS, _MAX_BATCH_MOLECULES, _HW_*

  Resumability, inherited from Stage 10.1 in full:
      _CHECKPOINT_EVERY_STEPS, _KEEP_EPOCH_CHECKPOINTS, _SEED

Every other knob -- the loss weights, K, top-k, temperature, the fallback
weight, the unlikelihood weight, which layers unfreeze, epochs, learning rate,
mask percent -- is Stage 10's, read from the same config entries.

Output directory and shared lineage
------------------------------------
config.STAGE10_SHARED_OUTPUT decides whether this writes to its own
STAGE10_2_DIR or joins Stage 10 and Stage 10.1 in STAGE10_SHARED_DIR, sharing
one continuable checkpoint. See stage10_lineage.py -- in shared mode a run
begun on the reference implementation can be finished here, and the provenance
of every epoch is recorded so a mixed-execution lineage stays reportable.

Usage
-----
  python stage10_2_hardware_tuned_batched_AMP_DDP_training.py
  python stage10_2_hardware_tuned_batched_AMP_DDP_training.py --test
  ... --hardware        print the machine profile AND every knob's RESOLVED
                        value, then exit. Run this first on a new box -- it is
                        the only way to see what "auto" became there.
  ... --speed off       parity with Stage 10 for this run
  ... --variant b       train 10b instead of config.STAGE10_VARIANT
  ... --epochs N        override config.STAGE10_NUM_EPOCHS for this run
  ... --workers 0       force serial scoring
  ... --no-batch        force the per-molecule forward and sampler
  ... --fresh           ignore any checkpoint and start over
  ... --shared / --separate   override config.STAGE10_SHARED_OUTPUT
  ... --limit none      uncapped final property pass

  torchrun --nproc_per_node=4 stage10_2_hardware_tuned_batched_AMP_DDP_training.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import warnings
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn.functional as F
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import config
import stage10_data_split as split
import stage10_lineage as lineage
import stage10_vanila_backpropagation_training as s10
import stage10_1_parallel_RDKit_scoring_resumable_training as s10_1
from hardware_autotune import get_profile
from stage9_masked_property_finetune import collect_all_training_pairs

# Execution machinery is shared with Stage 9.1 rather than copied: the pool's
# lifecycle (persistent spawn context, an initializer that pre-builds the
# PAINS/Brenk catalogs and pins BLAS threads, terminate-on-exit), the DDP
# rendezvous and the length-bucketed batcher are all stage-agnostic, and
# copying them is how two stages drift into meaning different things by
# "auto". Every one of them takes explicit arguments below, resolved from the
# STAGE10_2_* namespace, so Stage 9.1's config never leaks in here.
from stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning import (
    ScoringPool,
    build_batches,
    ddp_cleanup,
    ddp_setup,
    _all_reduce_mean,
    _shard_batches,
)

try:
    from tqdm import tqdm
except ImportError:                                   # pragma: no cover
    # A silent stand-in with the same call surface, tqdm.write included;
    # the old inline stub was a bare function and had no .write.
    from tqdm_compat import tqdm  # type: ignore[misc]


STAGE = "stage10_2"

# ── knobs re-exported from Stage 10, so there is exactly one definition ──────
# Listed explicitly rather than star-imported, so a rename in Stage 10 fails
# here loudly at import time instead of silently falling back to a default.
VARIANT             = s10.VARIANT
K_CANDIDATES        = s10.K_CANDIDATES
TOP_K               = s10.TOP_K
TEMPERATURE         = s10.TEMPERATURE
W_QED               = s10.W_QED
W_SA                = s10.W_SA
W_NOVELTY           = s10.W_NOVELTY
W_TOX_ALERT         = s10.W_TOX_ALERT
W_VALID             = s10.W_VALID
FALLBACK_WEIGHT     = s10.FALLBACK_WEIGHT
UNLIKELIHOOD_WEIGHT = s10.UNLIKELIHOOD_WEIGHT
NUM_EPOCHS          = s10.NUM_EPOCHS
LEARNING_RATE       = s10.LEARNING_RATE
GRAD_CLIP           = s10.GRAD_CLIP
MAX_MODEL_TOKENS    = s10.MAX_MODEL_TOKENS
S_WORST_VALID       = s10.S_WORST_VALID
LOSS_INVALID        = s10.LOSS_INVALID
LOSS_TERMS          = s10.LOSS_TERMS

# The loss composition itself comes from Stage 10.1, which already restated it
# to read a pre-computed measurement dict (the pool returns those). Importing
# it means the restatement exists once in the family, and Stage 10.1's
# self-test already pins it against Stage 10 to zero tolerance.
compose_stage10_loss = s10_1.compose_stage10_loss

# Resumability is Stage 10.1's, wholesale.
_rng_state    = s10_1._rng_state
_restore_rng  = s10_1._restore_rng
_InterruptGuard = s10_1._InterruptGuard
_ask_resume   = s10_1._ask_resume

SEED = (getattr(config, "STAGE10_2_SEED", None)
        or getattr(config, "STAGE10_1_SEED", 42))
CKPT_EVERY = getattr(config, "STAGE10_2_CHECKPOINT_EVERY_STEPS", None)
if CKPT_EVERY is None:
    CKPT_EVERY = getattr(config, "STAGE10_1_CHECKPOINT_EVERY_STEPS", 100)
KEEP_EPOCH_CKPTS = getattr(config, "STAGE10_2_KEEP_EPOCH_CHECKPOINTS", None)
if KEEP_EPOCH_CKPTS is None:
    KEEP_EPOCH_CKPTS = getattr(config, "STAGE10_1_KEEP_EPOCH_CHECKPOINTS", True)


# ════════════════════════════════════════════════════════════════════════════
#  THE KNOBS  --  argument > config.STAGE10_2_<KNOB> > config.STAGE10_2_SPEED
# ════════════════════════════════════════════════════════════════════════════
#
# Same three-level precedence, and the same reasoning, as Stage 9.1's. It is
# restated in this namespace rather than imported because Stage 9.1's resolver
# reads config.STAGE9_1_*, and one stage silently inheriting another stage's
# speed preset is precisely the confusion these knobs exist to remove.
#
# WHAT THE PRESETS MEAN HERE
#
#   "off"   Stage 10's execution exactly: per-molecule forward, per-molecule
#           sampler, serial scoring, fp32, TF32 explicitly disabled, no
#           bucketing. Bit-identical to Stage 10 and Stage 10.1.
#
#   "safe"  Only what cannot change WHICH candidates get sampled or what the
#           objective is: the RDKit pool, plus thread and pool tuning. RDKit is
#           a pure function of two strings, so running it in another process
#           cannot move a float -- Stage 10.1's self-test pins pooled == serial
#           to zero tolerance. Still bit-identical to Stage 10. The one honest
#           caveat is that a different CPU thread count can reorder BLAS
#           reductions, so a CPU-ONLY run may differ in the last bits.
#
#   "fast"  The default. Batched forward and batched sampling, pooled scoring,
#           AMP, TF32, length bucketing. NOT bit-identical -- see the module
#           docstring for why that matters more here than in Stage 9.1.
#
#   "auto"  Alias of "fast", with every "auto" knob sized by hardware_autotune.
#
# WHAT THE PRESET DELIBERATELY DOES NOT TOUCH
#   STAGE10_2_BATCH_SIZE. Batch size sets the optimizer-step count, so a preset
#   named "fast" that quietly quartered the number of updates would be
#   measuring something other than speed.

_PRESET_KEYS = (
    "batched_forward", "scoring_workers", "amp", "tf32", "matmul_precision",
    "torch_threads", "worker_blas_threads", "length_bucketing",
    "pool_start_method", "pool_chunk_factor", "pool_maxtasksperchild",
    "max_seq_tokens", "grad_scaler", "grad_scaler_init_scale",
    "ddp_split_batch", "ddp_find_unused", "ddp_backend",
    "hw_min_total_steps", "hw_max_batch", "hw_mem_headroom",
    "max_batch_tokens", "max_batch_molecules",
)

# Knobs that are NOT speedups, and so hold the same value at every level:
#   worker_blas_threads / pool_*  shape the pool, which only exists when
#                                 scoring_workers > 0 anyway
#   max_seq_tokens                the model's window
#   grad_scaler                   forced by the AMP dtype, not chosen for speed
#   ddp_split_batch               a CORRECTNESS property (equal optimizer-step
#                                 count across world sizes), not an optimisation
#   hw_* / max_batch_*            only consulted by the sizing modes
_INVARIANT = {
    "worker_blas_threads":    "auto",
    "pool_start_method":      "auto",
    "pool_chunk_factor":      2,
    "pool_maxtasksperchild":  None,
    "max_seq_tokens":         MAX_MODEL_TOKENS,
    "grad_scaler":            "auto",
    "grad_scaler_init_scale": None,
    "ddp_split_batch":        True,
    "ddp_find_unused":        False,
    "ddp_backend":            "auto",
    "hw_min_total_steps":     2000,
    "hw_max_batch":           512,
    "hw_mem_headroom":        0.55,
    "max_batch_tokens":       65536,
    "max_batch_molecules":    512,
}

_SPEED_PRESETS: Dict[str, dict] = {
    "off": dict(_INVARIANT,
                batched_forward=False, scoring_workers=0, amp=False,
                tf32=False, matmul_precision="highest", torch_threads=0,
                length_bucketing=False),
    "safe": dict(_INVARIANT,
                 batched_forward=False, scoring_workers="auto", amp=False,
                 tf32=False, matmul_precision="highest", torch_threads="auto",
                 length_bucketing=False),
    "fast": dict(_INVARIANT,
                 batched_forward="auto", scoring_workers="auto", amp="auto",
                 tf32=True, matmul_precision="auto", torch_threads="auto",
                 length_bucketing=True),
}
_SPEED_PRESETS["auto"] = dict(_SPEED_PRESETS["fast"])

# Set by --speed; overrides config.STAGE10_2_SPEED for this process, so two
# presets can be A/B'd without editing config.
_SPEED_OVERRIDE: Optional[str] = None


def resolve_speed(name: str = None) -> str:
    """
    The active preset name.

    An unknown name raises rather than falling back to "fast": a typo'd preset
    that quietly runs every speedup -- and so quietly stops being comparable to
    Stage 10 -- is exactly the surprise these knobs exist to remove.
    """
    if name is None:
        name = _SPEED_OVERRIDE or getattr(config, "STAGE10_2_SPEED", "fast")
    key = str(name).lower()
    if key not in _SPEED_PRESETS:
        raise ValueError(
            f"config.STAGE10_2_SPEED={name!r} is not a preset. "
            f"Choose one of {sorted(_SPEED_PRESETS)}."
        )
    return key


def hw_setting(knob: str, override=None, speed: str = None):
    """
    Resolve one hardware knob through the three-level precedence above.

    `override` is the caller's explicit value and wins outright; None means
    "not specified", which is why every knob's OFF state is spelled False or 0
    rather than None.
    """
    if knob not in _PRESET_KEYS:                     # pragma: no cover
        raise KeyError(f"unknown hardware knob {knob!r}")
    if override is not None:
        return override
    from_config = getattr(config, "STAGE10_2_" + knob.upper(), None)
    if from_config is not None:
        return from_config
    return _SPEED_PRESETS[resolve_speed(speed)][knob]


def _is_auto(v) -> bool:
    return isinstance(v, str) and v.lower() == "auto"


# ════════════════════════════════════════════════════════════════════════════
#  RESOLVING THE INDIVIDUAL KNOBS
# ════════════════════════════════════════════════════════════════════════════

def resolve_batched(setting=None) -> bool:
    """
    STAGE10_2_BATCHED_FORWARD / preset -> bool.

    "auto" means "on when CUDA is present": batching mainly recovers per-launch
    overhead, which dominates a batch-1 forward on GPU and matters much less on
    CPU. This is THE knob that decides whether this run is bit-identical to
    Stage 10, so it is reported first in every banner.
    """
    setting = hw_setting("batched_forward", setting)
    if _is_auto(setting):
        return torch.cuda.is_available()
    return bool(setting)


def resolve_workers(setting=None) -> int:
    """
    STAGE10_2_SCORING_WORKERS / preset -> worker count (0 = serial, no pool).

    "auto" defers to hardware_autotune, which reads the SLURM allocation, the
    CPU affinity mask and the cgroup quota rather than os.cpu_count() -- on a
    shared HPC node the host count is not what this job may use. Returns 0
    rather than 1 for a single worker: a one-worker pool adds pickling and IPC
    for no parallelism. That is the common case on a 2-vCPU Colab box.
    """
    setting = hw_setting("scoring_workers", setting)
    if _is_auto(setting):
        setting = get_profile().cpu_workers
    n = int(setting)
    return 0 if n <= 1 else n


def resolve_torch_threads(setting=None) -> int:
    """
    STAGE10_2_TORCH_THREADS / preset -> intra-op threads for THIS process.

    0 means "leave torch's own default alone", which is what "off" wants: not
    touching a global is not the same as setting it to the value it already
    held, and only the former survives a caller that set it deliberately.

    Note the interaction with the scoring pool: this process and its W workers
    share one CPU allocation, and with K=16 candidates per molecule the parent
    spends most of the step waiting on pool.map. On a small box the honest
    setting is FEWER threads here, not more.
    """
    setting = hw_setting("torch_threads", setting)
    if _is_auto(setting):
        return get_profile().torch_threads
    return max(0, int(setting))


def resolve_worker_blas_threads(setting=None) -> int:
    """
    STAGE10_2_WORKER_BLAS_THREADS / preset -> threads INSIDE each pool worker.

    "auto" is 1. W workers each starting T BLAS threads on a W-core allocation
    is the classic oversubscription that makes a pooled run slower than a
    serial one; raise it only if you have deliberately left cores idle.
    """
    setting = hw_setting("worker_blas_threads", setting)
    if _is_auto(setting):
        return 1
    return max(1, int(setting))


def resolve_pool_start_method(setting=None) -> str:
    """
    STAGE10_2_POOL_START_METHOD / preset -> "spawn" | "fork" | "forkserver".

    "auto" is "spawn", and that default is load-bearing rather than cautious:
    fork copies this process's CUDA context into every child, which corrupts it
    in ways that surface much later and elsewhere. "fork" is faster to start
    and is a legitimate choice on a CPU-only Linux run; it is refused on
    platforms that lack it, and downgraded with a warning once CUDA is live.

    "forkserver" is exempt from that downgrade: the server interpreter is
    started by fork+exec and so inherits no CUDA context, the workers fork from
    the server rather than from this process, and the main module is re-imported
    once instead of once per worker. On Colab that is the difference between
    paying torch + transformers W times and paying it once. Note that OpenMP is
    then already initialised when init_worker sets OMP_NUM_THREADS, so only its
    torch.set_num_threads call still bites.
    """
    setting = hw_setting("pool_start_method", setting)
    method = "spawn" if _is_auto(setting) else str(setting).lower()
    available = mp.get_all_start_methods()
    if method not in available:
        warnings.warn(
            f"start method {method!r} is unavailable on this platform "
            f"(have {available}); falling back to 'spawn'.",
            RuntimeWarning, stacklevel=2,
        )
        return "spawn"
    if method == "fork" and torch.cuda.is_initialized():
        warnings.warn(
            f"config.STAGE10_2_POOL_START_METHOD={method!r} with an initialised "
            f"CUDA context: forking copies that context into every worker and "
            f"corrupts it. Using 'spawn' instead -- 'forkserver' gives most of "
            f"fork's startup saving and is safe here.",
            RuntimeWarning, stacklevel=2,
        )
        return "spawn"
    return method


def resolve_pool_maxtasks(setting=None) -> int:
    """
    STAGE10_2_POOL_MAXTASKSPERCHILD / preset -> tasks before a worker is
    recycled, as an int, with 0 meaning "never recycle".

    The 0 rather than None is the point, and it is what keeps this stage's
    namespace sealed. ScoringPool resolves its own keyword arguments through
    Stage 9.1's hw_setting, which treats None as "not specified" and falls
    through to config.STAGE9_1_POOL_MAXTASKSPERCHILD. Passing None here would
    therefore make a Stage 9.1 setting silently govern a Stage 10.2 pool --
    exactly the cross-stage inheritance this file's knobs exist to prevent.
    0 is an explicit value, so the override wins, and ScoringPool maps it back
    to None (its own "never recycle") because 0 is falsy.
    """
    setting = hw_setting("pool_maxtasksperchild", setting)
    return 0 if not setting else int(setting)


def resolve_max_seq_tokens(setting=None) -> int:
    """
    STAGE10_2_MAX_SEQ_TOKENS / preset -> truncation length for the padded
    forward.

    A memory knob more than a speed one: the padded cost is B x L x V, so
    halving L halves the logits tensor. Setting it below the longest pair in
    the data silently truncates molecules -- and here a truncated molecule also
    loses mask positions, so parent_target_ids will refuse to align it and the
    molecule is skipped. Reach for this only when a batch genuinely will not
    fit.
    """
    return max(8, int(hw_setting("max_seq_tokens", setting)))


def resolve_ddp_backend(setting=None):
    return hw_setting("ddp_backend", setting)


def resolve_amp(device: str, setting=None):
    """
    STAGE10_2_AMP -> a torch dtype for autocast, or None for full fp32.

    "auto" prefers bf16 where the hardware supports it (Ampere and later)
    because it needs no gradient scaler, and falls back to fp16 on Turing --
    which is what a standard Colab GPU runtime gives you (T4). Always None on
    CPU, where autocast buys nothing here and would only complicate the parity
    self-test.
    """
    setting = hw_setting("amp", setting)
    if not setting or _device_type(device) != "cuda":
        return None
    if isinstance(setting, str):
        s = setting.lower()
        if s == "bf16":
            return torch.bfloat16
        if s == "fp16":
            return torch.float16
        if s == "auto":
            return get_profile().amp_dtype
    return None


def amp_label(amp_dtype) -> Optional[str]:
    """The autocast dtype as the short string the provenance log stores."""
    if amp_dtype is None:
        return None
    return {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(
        amp_dtype, str(amp_dtype).replace("torch.", ""))


def _device_type(device: str) -> str:
    """"cuda:3" -> "cuda". autocast and GradScaler take a device TYPE, not an
    indexed device, and under DDP every rank past 0 holds an indexed one."""
    return str(device).split(":")[0]


def _autocast(device: str, amp_dtype):
    """autocast when a mixed-precision dtype is active, otherwise a no-op."""
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=_device_type(device), dtype=amp_dtype)


def apply_hardware_settings(log=None) -> Dict[str, object]:
    """
    Push the resolved CPU/GPU settings into torch -- once -- and report what
    actually landed.

    THIS IS THE ONLY PLACE IN STAGE 10.2 THAT WRITES A TORCH BACKEND GLOBAL,
    and that is the point. In Stage 9.1 those writes were once split between
    two functions that disagreed: hw.apply() switched TF32 on from the compute
    capability alone, and the config check that followed could only ever turn
    it ON, never off -- so `TF32 = False` printed "TF32 OFF" over kernels that
    were still using it. Here config decides and the hardware only fills in the
    "auto"s, so the banner and the kernels cannot disagree.

    Returns the values the banner should report -- resolved, not requested.
    """
    hw = get_profile()
    want_tf32 = bool(hw_setting("tf32")) and torch.cuda.is_available()
    prec = hw_setting("matmul_precision")
    if _is_auto(prec):
        # "high" is what auto-TF32 has always implied; "highest" is its honest
        # opposite, and leaving the global untouched would not be.
        prec = "high" if want_tf32 else "highest"

    applied = hw.apply(
        threads          = resolve_torch_threads(),
        tf32             = want_tf32,
        matmul_precision = prec,
    )
    if log is not None:
        for line in applied:
            log(f"  hardware: {line}")
    return {
        "tf32":             want_tf32 and hw.supports_tf32,
        "tf32_requested":   bool(hw_setting("tf32")),
        "matmul_precision": prec,
        "applied":          applied,
    }


def describe_hardware_settings(speed: str = None) -> Dict[str, object]:
    """
    Every knob's RESOLVED value, for the banner, for --hardware and for the
    self-test.

    Resolved, not configured: this is what the run will actually do after
    "auto" has been turned into a number by hardware_autotune. It is the answer
    to "which speedups did I allow, and am I still comparable to Stage 10?",
    which a printout of config alone cannot give.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batched = resolve_batched()
    return {
        "speed":               resolve_speed(speed),
        "batched_forward":     batched,
        "scoring_workers":     resolve_workers(),
        "amp":                 hw_setting("amp"),
        "amp_resolved":        amp_label(resolve_amp(device)),
        "tf32":                bool(hw_setting("tf32")),
        "matmul_precision":    hw_setting("matmul_precision"),
        "torch_threads":       resolve_torch_threads(),
        "worker_blas_threads": resolve_worker_blas_threads(),
        "length_bucketing":    bool(hw_setting("length_bucketing")),
        "pool_start_method":   resolve_pool_start_method(),
        "pool_chunk_factor":   int(hw_setting("pool_chunk_factor")),
        "max_seq_tokens":      resolve_max_seq_tokens(),
        "grad_scaler":         hw_setting("grad_scaler"),
        "ddp_split_batch":     bool(hw_setting("ddp_split_batch")),
        "ddp_find_unused":     bool(hw_setting("ddp_find_unused")),
        "ddp_backend":         resolve_ddp_backend(),
        "batch_size":          getattr(config, "STAGE10_2_BATCH_SIZE", None)
                               or s10.BATCH_SIZE,
        "stage10_identical":   (not batched
                                and not bool(hw_setting("length_bucketing"))
                                and resolve_amp(device) is None),
        "shared_output":       lineage.shared_enabled(),
        "output_dir":          lineage.resolve_save_dir(STAGE, VARIANT),
    }


def resolve_grad_scaler(amp_dtype, setting=None) -> bool:
    """
    STAGE10_2_GRAD_SCALER -> whether to wrap the backward in a GradScaler.

    "auto" means "exactly when the dtype is fp16". bf16 keeps fp32's exponent
    range, so its gradients do not underflow and a scaler would only add work
    and a failure mode. This is forced by the dtype rather than chosen for
    speed, which is why it sits in _INVARIANT.
    """
    setting = hw_setting("grad_scaler", setting)
    if _is_auto(setting):
        return amp_dtype == torch.float16
    return bool(setting)


# ════════════════════════════════════════════════════════════════════════════
#  BATCHING  --  deterministic, and addressable by index for resume
# ════════════════════════════════════════════════════════════════════════════

def _is_auto_batch(batch_size) -> bool:
    return isinstance(batch_size, str) and batch_size.lower() == "auto"


def batches_for_epoch(
    pairs:       List[Tuple[str, str]],
    tokenizer,
    epoch:       int,
    batch_size,
    seed:        int  = SEED,
    bucketing:   bool = None,
    max_tokens:  int  = None,
    max_mols:    int  = None,
) -> List[List[int]]:
    """
    The epoch's batches as lists of INDICES into `pairs`.

    Indices, not pairs, and a pure function of its arguments -- both because
    mid-epoch resume addresses a batch by its position in this list, so the
    list has to be reconstructible without having run the preceding epochs.

    Two regimes:

    NO BUCKETING AND A FIXED INTEGER BATCH SIZE. Delegates to Stage 10.1's
    _batches_for_epoch, which reproduces Stage 10's cumulative in-place shuffle
    exactly. That is what makes `--speed off` train on the same molecules in
    the same batches in the same order as Stage 10, and so what makes the
    parity assertion in the self-test meaningful.

    BUCKETING OR "auto" TOKEN PACKING. Delegates to Stage 9.1's build_batches,
    which sorts by token length, packs, then reshuffles the batch ORDER so the
    gradient sequence stays stochastic rather than becoming a length
    curriculum. Reusing it rather than restating it means "bucketed" cannot
    come to mean two different things in two stages.

    The trick that lets the shared batcher return indices: build_batches only
    ever reads the FIRST element of each pair (to measure token length) and
    otherwise moves pairs around opaquely. Passing (masked, index) tuples
    therefore gets the identical batching with the index carried through.
    """
    bucketing = bool(hw_setting("length_bucketing", bucketing))
    auto = _is_auto_batch(batch_size)

    if not bucketing and not auto:
        return s10_1._batches_for_epoch(len(pairs), epoch, int(batch_size), seed)

    if max_tokens is None:
        max_tokens = int(hw_setting("max_batch_tokens"))
    if max_mols is None:
        max_mols = int(hw_setting("max_batch_molecules"))

    tagged = [(masked, i) for i, (masked, _) in enumerate(pairs)]
    # Seeded per epoch so the permutation is an address, not a history: batch
    # 812 of epoch 3 is the same batch whether it is reached by running epochs
    # 1 and 2 or by resuming straight into it.
    rng = random.Random(seed * 1000 + epoch)
    built = build_batches(tagged, tokenizer, batch_size, rng,
                          bucketing=bucketing, max_tokens=max_tokens,
                          max_mols=max_mols)
    return [[i for _, i in b] for b in built]


# ════════════════════════════════════════════════════════════════════════════
#  ONE TRAINING STEP  --  batched forward, batched sampling, pooled scoring
# ════════════════════════════════════════════════════════════════════════════

def stage10_2_batch_loss(
    batch:       List[Tuple[str, str]],
    tokenizer,
    model,
    device:      str,
    pool:        "ScoringPool",
    variant:     str   = None,
    k_cand:      int   = None,
    top_k:       int   = None,
    temperature: float = None,
    fallback_weight:     float = None,
    unlikelihood_weight: float = None,
    batched:     bool  = None,
    amp_dtype          = None,
    max_tokens:  int   = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Stage 10's batch loss with the forward, the sampler and the scoring all
    lifted out of the per-molecule loop. Returns the identical (loss, stats)
    pair.

    With `batched=False` this delegates to Stage 10.1 -- one forward per
    molecule, one sampler call per molecule, pooled scoring -- and is therefore
    bit-identical to Stage 10. Everything below is the `batched=True` path.

    Three phases, so the pool sees the whole batch at once:

        phase 1  ONE padded [B, L] forward; ONE multinomial over every masked
                 position in the batch; decode all B x K candidates in one
                 batch_decode                                        (GPU)
        phase 2  measure all B x K candidates                        (pool)
        phase 3  compose, select best per molecule, build the CE     (here)

    WHAT IS SAFE HERE AND WHAT IS NOT
    ----------------------------------
    Phases 2 and 3 are safe to reorder in any execution: nothing in them
    touches a random number or a model weight. compute_property_components is a
    pure function of (generated, parent), pool.map preserves input order, and
    the arithmetic runs on the same values in the same sequence. That is why
    Stage 10.1 can be bit-identical while still using the pool.

    Phase 1 is NOT order-preserving, and this is the honest cost of this file.
    Stage 10 draws one multinomial per molecule; this draws one for every mask
    in the batch together. Same distribution, different sample -- like dealing
    from an identically shuffled deck in a different order. Because the sample
    SELECTS the training target, a different draw means a different target
    sequence, not merely a different gradient.

    Dropout compounds it: it draws once per FORWARD, so B separate forwards
    consume B masks and this one consumes one. That shifts the mask logits by
    ~1.2 on this model, dwarfing anything AMP contributes, and no seed can
    align it. Set STAGE10_2_BATCHED_FORWARD = False (or --speed off/safe) when
    the point of the run is parity rather than wall clock.

    PADDING CANNOT CONTAMINATE ANYTHING. Pad tokens are not mask tokens, so
    they are never selected by the mask index; the attention mask enters the
    encoder as an additive -inf bias; and the loss only ever reads rows at
    masked positions. Right padding is REQUIRED and checked, because left
    padding shifts RoBERTa's position ids whenever the pad id and
    model.config.pad_token_id disagree, and corrupts real tokens silently.
    """
    variant             = (variant or VARIANT).lower()
    k_cand              = K_CANDIDATES        if k_cand is None else k_cand
    top_k               = TOP_K               if top_k is None else top_k
    temperature         = TEMPERATURE         if temperature is None else temperature
    fallback_weight     = FALLBACK_WEIGHT     if fallback_weight is None else fallback_weight
    unlikelihood_weight = UNLIKELIHOOD_WEIGHT if unlikelihood_weight is None else unlikelihood_weight
    batched             = resolve_batched() if batched is None else bool(batched)

    if not batched:
        # The parity path. Stage 10.1's loss IS Stage 10's, to the bit.
        return s10_1.stage10_1_batch_loss(
            batch, tokenizer, model, device, pool, variant=variant,
            k_cand=k_cand, top_k=top_k, temperature=temperature,
            fallback_weight=fallback_weight,
            unlikelihood_weight=unlikelihood_weight)

    if tokenizer.padding_side != "right":
        raise ValueError(
            f"tokenizer.padding_side must be 'right' for the batched forward, "
            f"got {tokenizer.padding_side!r}.")

    stats: Dict[str, float] = {
        "n": 0.0, "fallback": 0.0, "cand_valid": 0.0, "cand_total": 0.0,
        "best_loss": 0.0, "unlikelihood": 0.0, "skipped": 0.0, "best_valid": 0.0,
    }
    for key in LOSS_TERMS:
        stats[key] = 0.0

    # ── phase 1a: one padded forward for the whole batch ──────────────────
    # STAGE10_2_MAX_SEQ_TOKENS is a CEILING, never a raise: a tokenizer that
    # reports a shorter window than the knob still wins, because exceeding the
    # model's own positional range is a crash, not a slow run.
    cap = resolve_max_seq_tokens(max_tokens)
    tok_max = getattr(tokenizer, "model_max_length", cap)
    if tok_max is None or tok_max > 1024:
        tok_max = cap
    max_len = min(int(tok_max), cap)

    masked_list = [m for m, _ in batch]
    parents     = [p for _, p in batch]
    enc = tokenizer(masked_list, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len).to(device)
    ids       = enc["input_ids"]                        # [B, L]
    attn_mask = enc["attention_mask"]                   # [B, L]
    B, L = ids.shape

    with _autocast(device, amp_dtype):
        out = model(input_ids=ids, attention_mask=attn_mask)

    # Flat index of every masked position across the batch. Padding is never
    # selected: pad tokens are not mask tokens.
    b_idx, l_idx = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if b_idx.numel() == 0:
        stats["skipped"] += float(B)
        return torch.zeros((), device=device, requires_grad=True), stats

    # .float() before any softmax: under fp16 autocast a log_softmax over a
    # 767-wide vocabulary loses meaningful precision, and these values feed
    # both the selection and the gradient. The cast touches only the masked
    # rows, not the whole [B, L, V], so it is negligible beside the forward.
    sel = out.logits[b_idx, l_idx].float()              # [M, V], carries grad
    M, V = sel.shape

    # ── phase 1b: ONE multinomial for every mask in the batch ─────────────
    # Same top-k + temperature construction as s10._sample_candidates, applied
    # to M rows at once instead of one molecule's n_masks rows at a time.
    scaled = (sel / max(temperature, 1e-8)).detach()
    probs  = F.softmax(scaled, dim=-1)
    kk     = min(top_k, V)
    top    = torch.topk(probs, k=kk, dim=-1)
    renorm = top.values / top.values.sum(dim=-1, keepdim=True)
    picks  = torch.multinomial(renorm, num_samples=k_cand, replacement=True)
    cand_ids = top.indices.gather(1, picks)             # [M, K]

    # ── phase 1c: decode all B x K candidates in one call ─────────────────
    # filled[b, j, :] is molecule b with candidate j written into its masks.
    # The advanced indices sit on dims 0 and 2 with a slice between them, so
    # the broadcast result is [M, K] -- exactly cand_ids' shape.
    filled = ids.unsqueeze(1).repeat(1, k_cand, 1)      # [B, K, L]
    filled[b_idx, :, l_idx] = cand_ids
    decoded = tokenizer.batch_decode(filled.view(B * k_cand, L),
                                     skip_special_tokens=True)
    flat = [(s.replace(" ", ""), parents[i // k_cand])
            for i, s in enumerate(decoded)]

    # ── phase 2: every candidate in the batch, measured at once ───────────
    measured = pool.measure(flat)

    # ── phase 3: compose, select, differentiate ───────────────────────────
    log_probs_all = F.log_softmax(sel, dim=-1)          # [M, V], carries grad
    # Rows of the flat mask index belonging to each molecule, in order. b_idx
    # is produced by nonzero(), which scans row-major, so each molecule's rows
    # are already contiguous and ascending in l_idx -- i.e. in mask order,
    # which is what parent_target_ids returns its tokens in.
    rows_by_mol: List[List[int]] = [[] for _ in range(B)]
    for row, b in enumerate(b_idx.tolist()):
        rows_by_mol[b].append(row)

    total = torch.zeros((), device=device)
    n_weighted = 0.0

    for b in range(B):
        rows = rows_by_mol[b]
        if not rows:
            stats["skipped"] += 1
            continue
        row_idx = torch.tensor(rows, device=device, dtype=torch.long)
        cands = cand_ids[row_idx].t().contiguous()      # [K, n_masks]

        best_loss, best_ids, best_comps = None, None, None
        invalid_rows: List[torch.Tensor] = []
        for j in range(k_cand):
            loss_c, comps = compose_stage10_loss(measured[b * k_cand + j])
            stats["cand_total"] += 1
            if comps["valid"]:
                stats["cand_valid"] += 1
            else:
                invalid_rows.append(cands[j])
            if best_loss is None or loss_c < best_loss:
                best_loss, best_ids, best_comps = loss_c, cands[j], comps

        used_fallback = (best_comps is None) or (not best_comps["valid"])
        if used_fallback:
            tgt = s10.parent_target_ids(masked_list[b], parents[b], tokenizer)
            if tgt is None or len(tgt) != len(rows):
                stats["skipped"] += 1
                continue
            best_ids = torch.tensor(tgt, device=device, dtype=torch.long)
            weight = fallback_weight
            stats["fallback"] += 1
        else:
            weight = 1.0

        lp = log_probs_all[row_idx]                     # [n_masks, V]
        ce = F.nll_loss(lp, best_ids, reduction="mean")
        mol_loss = weight * ce

        if variant == "a" and invalid_rows and unlikelihood_weight > 0:
            bad = torch.stack(invalid_rows)             # [n_bad, n_masks]
            keep = bad != best_ids.unsqueeze(0)
            if keep.any():
                p = lp.exp()
                p_bad = p.gather(1, bad.t()).t()
                u = -torch.log((1.0 - p_bad).clamp(min=1e-6))
                unlike = (u * keep).sum() / keep.sum()
                mol_loss = mol_loss + unlikelihood_weight * unlike
                stats["unlikelihood"] += float(unlike.detach())

        total = total + mol_loss
        n_weighted += 1.0
        stats["n"] += 1
        stats["best_loss"] += best_loss
        stats["best_valid"] += 1.0 if (best_comps and best_comps["valid"]) else 0.0
        for key in LOSS_TERMS:
            stats[key] += best_comps[key] if best_comps else 0.0

    if n_weighted == 0:
        return torch.zeros((), device=device, requires_grad=True), stats
    return total / n_weighted, stats


# ════════════════════════════════════════════════════════════════════════════
#  CHECKPOINTING  --  Stage 10.1's, through the shared lineage module
# ════════════════════════════════════════════════════════════════════════════

def _fingerprint(variant: str, batch_size, k_cand: int, num_epochs: int,
                 n_pairs: int, seed: int, bucketing: bool) -> Dict[str, object]:
    """
    The settings a mid-epoch resume depends on.

    batch_index is an offset into a list built by batches_for_epoch, so it only
    means anything if n_pairs, batch_size, seed AND bucketing are what they
    were when it was written. Bucketing is in here and is not in Stage 10.1's
    fingerprint, because it is the extra thing that can change batch
    COMPOSITION here: flip it mid-run and batch 812 holds different molecules.
    """
    return {"variant": variant, "batch_size": batch_size, "k_cand": k_cand,
            "num_epochs": num_epochs, "n_pairs": n_pairs, "seed": seed,
            "bucketing": bool(bucketing)}


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_stage10_2_training(
    pairs:       List[Tuple[str, str]],
    save_dir:    str   = None,
    variant:     str   = None,
    num_epochs:  int   = NUM_EPOCHS,
    batch_size         = None,
    lr:          float = LEARNING_RATE,
    grad_clip:   float = GRAD_CLIP,
    k_cand:      int   = K_CANDIDATES,
    workers:     int   = None,
    batched:     bool  = None,
    ckpt_every:  int   = None,
    seed:        int   = SEED,
    fresh:       bool  = False,
    auto_resume: Optional[bool] = None,
    speed:       str   = None,
) -> Dict[str, list]:
    """
    Stage 10's supervised best-of-K loop, hardware-tuned end to end.

    Returns the same history dict Stage 10 returns, so _plot_history and every
    downstream reader work unchanged.

    `auto_resume` overrides the interactive prompt: True resumes silently,
    False ignores the checkpoint. Left as None it asks, falling back to
    resuming when stdin is not a terminal.
    """
    if speed is not None:
        # Process-wide, exactly as Stage 9.1's --speed is, and for the same
        # reason: batches_for_epoch, ScoringPool and the loss all read the
        # resolvers directly. A `speed=` that only relabelled the banner while
        # the run used config's preset would be worse than no argument at all.
        global _SPEED_OVERRIDE
        _SPEED_OVERRIDE = resolve_speed(speed)

    variant = (variant or VARIANT).lower()
    if variant not in ("a", "b"):
        raise ValueError(f"variant must be 'a' or 'b', got {variant!r}")
    save_dir = save_dir or lineage.resolve_save_dir(STAGE, variant)

    workers = (resolve_workers(getattr(config, "STAGE10_2_SCORING_WORKERS", None))
               if workers is None else resolve_workers(workers))
    batched = resolve_batched() if batched is None else bool(batched)
    bucketing = bool(hw_setting("length_bucketing"))
    if ckpt_every is None:
        ckpt_every = CKPT_EVERY
    if batch_size is None:
        batch_size = getattr(config, "STAGE10_2_BATCH_SIZE", None) or s10.BATCH_SIZE

    # ── multi-GPU rendezvous (a no-op unless launched under torchrun) ─────
    rank, world_size, local_rank, is_dist = ddp_setup(
        backend=resolve_ddp_backend())
    is_main = (rank == 0)

    def log(msg: str) -> None:
        """Only rank 0 prints, or N ranks interleave the same lines N times."""
        if is_main:
            tqdm.write(msg)

    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    hw = get_profile()
    # The only call that touches torch's thread count, TF32 flag and matmul
    # precision -- see apply_hardware_settings for why that is enforced.
    precision = apply_hardware_settings(log=log)

    # ── hold out the validation fold BEFORE the fingerprint ───────────────
    # Whole Bemis-Murcko groups, from the manifest every Stage 10 variant
    # shares, so len(pairs) -- and therefore the resume fingerprint and the
    # DDP shard count -- refers to the TRAINING fold only.
    pairs, val_pairs, _ = split.split_pairs(pairs, log=log)
    if is_main:
        for line in split.describe(
                split.get_split([q for _, q in pairs + val_pairs], log=log),
                (pairs, val_pairs, [])):
            log(line)
    best_val = float("inf")

    if isinstance(batch_size, str) and batch_size.lower() == "hardware":
        # Sized from the dataset and epoch budget, NOT from free GPU memory:
        # at 44M parameters the backbone cannot fill a modern card, so memory
        # is the wrong target. See HardwareProfile.recommended_batch_size.
        batch_size = hw.recommended_batch_size(
            seq_len=128, vocab=767, n_pairs=len(pairs), n_epochs=num_epochs,
            min_total_steps = int(hw_setting("hw_min_total_steps")),
            max_batch       = int(hw_setting("hw_max_batch")),
            headroom        = float(hw_setting("hw_mem_headroom")),
        )
        log(f"  Batch size {batch_size} chosen by hardware_autotune "
            f"(config.STAGE10_2_BATCH_SIZE='hardware', "
            f">= {int(hw_setting('hw_min_total_steps'))} total optimizer steps, "
            f"cap {int(hw_setting('hw_max_batch'))}).")

    # The configured batch is the GLOBAL batch; split it so the optimizer-step
    # count matches a single-GPU run and the DDP gain is real wall clock.
    global_batch = batch_size
    split_batch = bool(hw_setting("ddp_split_batch"))
    if is_dist and split_batch and not isinstance(batch_size, str):
        batch_size = max(1, int(batch_size) // world_size)
        log(f"  DDP: global batch {global_batch} split across {world_size} "
            f"rank(s) -> {batch_size} per rank (step count unchanged).")
    elif is_dist and not split_batch:
        log(f"  DDP: STAGE10_2_DDP_SPLIT_BATCH=False, so {batch_size} is PER "
            f"RANK. Effective batch is {world_size}x larger and this run takes "
            f"{world_size}x FEWER optimizer steps than a single-GPU run on the "
            f"same data. Raise the learning rate or the epoch count to match.")

    history: Dict[str, list] = s10.new_history()

    def _fresh_agg() -> Dict[str, float]:
        a = {k: 0.0 for k in ("loss", "best_loss", "fallback", "cand_valid",
                              "cand_total", "n", "unlikelihood", "best_valid")}
        for key in LOSS_TERMS:
            a[key] = 0.0
        return a

    fp = _fingerprint(variant, global_batch, k_cand, num_epochs, len(pairs),
                      seed, bucketing)

    # ── model first: the AMP dtype is part of this run's identity, and the
    #    provenance banner has to name it before anything is restored ───────
    tokenizer, model, device = s10.load_model_last_layers()
    if is_dist and torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        model.to(device)
    amp_dtype = resolve_amp(device)
    profile = lineage.execution_profile(
        STAGE, speed=resolve_speed(speed), amp=amp_label(amp_dtype),
        batched=batched, workers=workers, device=_device_type(device),
        world_size=world_size, bucketing=bucketing)

    # ── resume ────────────────────────────────────────────────────────────
    ckpt = None if fresh else lineage.load_state(save_dir, STAGE, log=log)
    start_epoch, start_batch, global_step = 1, 0, 0
    agg, n_steps = _fresh_agg(), 0
    provenance: List[dict] = []
    resume_state = None

    # Under DDP the prompt is skipped entirely rather than asked on rank 0:
    # ranks that disagreed about resuming would train different weights and
    # all-reduce them together, which is worse than either answer. Resuming is
    # also the non-destructive choice -- "no" starts over and the first save
    # overwrites the checkpoint.
    def _want_resume() -> bool:
        if auto_resume is not None:
            return bool(auto_resume)
        if is_dist:
            log("  Distributed run -- resuming automatically (pass --fresh to "
                "start over instead).")
            return True
        return _ask_resume(save_dir, ckpt)

    if ckpt and _want_resume():
        banner = lineage.lineage_banner(ckpt, profile)
        if banner:
            log(banner)
        old_fp = ckpt.get("fingerprint", {})
        # batch_index addresses a list built from these four numbers. If any
        # changed, the index points at different molecules, so the honest move
        # is to keep the weights and the optimizer and restart the epoch --
        # not to silently train on the wrong slice.
        batching_same = all(old_fp.get(k) == fp[k]
                            for k in ("n_pairs", "batch_size", "seed", "bucketing"))
        start_epoch = int(ckpt["epoch"])
        global_step = int(ckpt["global_step"])
        history     = s10.ensure_history_keys(ckpt["history"])
        provenance  = ckpt.get("provenance") or []
        resume_state = ckpt
        if batching_same:
            start_batch = int(ckpt["batch_index"])
            agg     = ckpt.get("agg") or _fresh_agg()
            n_steps = int(ckpt.get("n_steps", 0))
        else:
            changed = [k for k in ("n_pairs", "batch_size", "seed", "bucketing")
                       if old_fp.get(k) != fp[k]]
            # Only worth saying when it actually costs something. A checkpoint
            # written at an epoch BOUNDARY is already at batch 0, so nothing is
            # discarded and the warning would be pure noise -- which is the
            # common case in shared mode, where Stage 10 hands over at every
            # boundary and never buckets.
            if int(ckpt["batch_index"]) != 0:
                log(f"  Batching settings changed since the checkpoint "
                    f"({', '.join(changed)}); the saved batch index no longer "
                    f"addresses the same molecules, so the "
                    f"{ckpt['batch_index']} batches already done in epoch "
                    f"{start_epoch} are repeated. Weights and optimizer are "
                    f"kept.")
            start_batch, agg, n_steps = 0, _fresh_agg(), 0
        for k in ("variant", "k_cand"):
            if old_fp.get(k) != fp[k]:
                log(f"  NOTE: {k} changed "
                    f"({old_fp.get(k)!r} -> {fp[k]!r}) since the checkpoint.")
        log(f"\n  Resuming at epoch {start_epoch}, batch {start_batch}, "
            f"step {global_step}"
            + (f" (checkpoint written by {ckpt.get('written_by')})"
               if ckpt.get("written_by") and ckpt.get("written_by") != STAGE
               else "") + ".")
    else:
        log("\n  Starting a fresh training run.")

    n_batches_per_epoch = len(batches_for_epoch(
        pairs, tokenizer, max(start_epoch, 1), batch_size, seed, bucketing))
    if start_epoch > num_epochs or (start_epoch == num_epochs
                                    and start_batch >= n_batches_per_epoch):
        log("  Training already complete (all epochs done).")
        ddp_cleanup(is_dist)
        return history

    if resume_state is not None:
        # strict=False by necessity: the checkpoint holds ONLY the unfrozen
        # tensors, so every frozen weight is reported missing and is supposed
        # to be -- it came from the pretrained model.
        model.load_state_dict(resume_state["trainable"], strict=False)
        log(f"  Restored {len(resume_state['trainable'])} trainable "
            f"tensor(s) from the checkpoint.")
    model.train()

    core = model                                  # unwrapped, for save/eval
    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=bool(hw_setting("ddp_find_unused")),
        )

    optimizer = torch.optim.Adam(
        [p for p in core.parameters() if p.requires_grad], lr=lr)
    if resume_state is not None and resume_state.get("optimizer"):
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
            log("  Optimizer state restored (Adam moments preserved).")
        except Exception as e:
            log(f"  Could not restore optimizer state ({e}); "
                f"continuing with a fresh Adam.")
    _restore_rng(resume_state.get("rng") if resume_state else None)
    if resume_state is None:
        # Seed the candidate-sampling stream so a fresh run is reproducible.
        # Rank-offset under DDP: identical streams on every rank would have all
        # ranks draw the same candidates for different molecules, which is not
        # wrong but wastes the extra exploration multiple ranks could give.
        torch.manual_seed(seed + rank)
    del resume_state

    scaler = None
    if resolve_grad_scaler(amp_dtype) and _device_type(device) == "cuda":
        init_scale = hw_setting("grad_scaler_init_scale")
        scaler = torch.amp.GradScaler(
            _device_type(device),
            **({} if init_scale is None else {"init_scale": float(init_scale)}),
        )

    log(
        f"\n  Variant {variant} : "
        + ("CE toward best candidate + unlikelihood on invalid ones"
           if variant == "a" else "CE toward best candidate only")
        + f"\n  K candidates    : {k_cand}"
        + f"\n  batch size      : {batch_size}"
        + (f" per rank ({global_batch} global)" if is_dist and split_batch else "")
        + f"  ({n_batches_per_epoch} batches/epoch)"
        + f"\n  speed preset    : {resolve_speed(speed)}"
          f"   [config.STAGE10_2_SPEED]"
        + f"\n  forward/sampler : "
        + ("BATCHED -- one padded forward and one multinomial per batch. "
           "NOT bit-identical to Stage 10 (a different candidate can win "
           "best-of-K); compare distributionally."
           if batched else
           "per-molecule -- bit-identical to Stage 10 and Stage 10.1.")
        + f"\n  scoring workers : "
        + (f"{workers}  ({k_cand * (batch_size if isinstance(batch_size, int) else 0)}"
           f" candidates/batch across the pool)" if workers else "serial (no pool)")
        + f"\n  precision       : "
        + (f"{amp_label(amp_dtype)} autocast (fp32 log-softmax/CE)"
           if amp_dtype else "fp32")
        + (" + GradScaler" if scaler is not None else "")
        + f"\n  TF32 matmul     : "
        + ("ON" if precision["tf32"] else
           ("OFF" if precision["tf32_requested"] else "OFF (not requested)"))
        + ("  -- requested but this GPU has no TF32 (needs compute capability 8.0+)"
           if precision["tf32_requested"] and not precision["tf32"] else "")
        + f"\n  length bucketing: {'ON' if bucketing else 'OFF'}"
        + f"\n  checkpoint      : "
        + (f"every {ckpt_every} steps + every epoch" if ckpt_every
           else "every epoch only (mid-epoch saves disabled)")
        + f"\n  hardware        : {hw.environment}, {hw.cpu_count} CPU "
          f"({hw.cpu_source}), device {device}"
        + (f", {world_size} ranks" if is_dist else "")
        + f"\n  loss (valid)    : {W_QED}*(1-QED) + {W_SA}*(SA-1)/9 "
          f"+ {W_NOVELTY}*similarity + {W_TOX_ALERT}*alert   -> at most {S_WORST_VALID}"
        + f"\n  loss (invalid)  : {LOSS_INVALID}  (= {S_WORST_VALID} + w_valid {W_VALID})"
        + f"\n  fallback weight : {FALLBACK_WEIGHT}  (parent-reconstruction steps)"
    )

    total_remaining = ((num_epochs - start_epoch + 1) * n_batches_per_epoch
                       - start_batch)
    # A live bar on a TTY, a status line every ckpt_every batches when there is
    # no TTY to draw one on (Colab's `!python`, nohup, a piped log). disable
    # keeps every rank but the main one silent, as it did for the bar. See
    # stage10_lineage.Progress.
    pbar = lineage.Progress(total=max(total_remaining, 0),
                            desc=f"Stage 10.2{variant} training",
                            every=ckpt_every or 100, disable=not is_main)

    stopped_early = False
    epoch = start_epoch

    pool_kwargs = dict(
        start_method     = resolve_pool_start_method(),
        chunk_factor     = int(hw_setting("pool_chunk_factor")),
        maxtasksperchild = resolve_pool_maxtasks(),
        blas_threads     = resolve_worker_blas_threads(),
    )

    with ScoringPool(workers, **pool_kwargs) as pool, _InterruptGuard() as guard:
        for epoch in range(start_epoch, num_epochs + 1):
            batches = batches_for_epoch(pairs, tokenizer, epoch, batch_size,
                                        seed, bucketing)
            first = start_batch if epoch == start_epoch else 0
            if first == 0:
                agg, n_steps = _fresh_agg(), 0

            # Ranks must agree on the step count BEFORE the loop, or a rank
            # that runs out early stops all-reducing and every other rank
            # blocks forever. _shard_batches takes the MIN across ranks.
            #
            # ONE APPROXIMATION, STATED: the checkpoint records rank 0's
            # cursor, and re-sharding from it reassigns which rank owns which
            # batch. A MID-EPOCH resume under DDP therefore repeats at most
            # world_size - 1 batches that non-zero ranks had already done --
            # extra updates on already-seen data, never skipped data. Resuming
            # at an epoch BOUNDARY (batch_index = 0) is exact, and so is every
            # resume on a single device. Set STAGE10_2_CHECKPOINT_EVERY_STEPS
            # = 0 if you would rather a multi-GPU run only ever resume at
            # boundaries.
            todo = list(range(first, len(batches)))
            todo = _shard_batches([[i] for i in todo], rank, world_size,
                                  is_dist, device)
            todo = [t[0] for t in todo]

            for bi in todo:
                batch = [pairs[i] for i in batches[bi]]
                optimizer.zero_grad()
                loss, stats = stage10_2_batch_loss(
                    batch, tokenizer, model, device, pool, variant=variant,
                    k_cand=k_cand, batched=batched, amp_dtype=amp_dtype)

                if stats["n"] == 0:
                    pbar.update(1)
                    continue

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        # unscale_ first, or the clip threshold is applied to
                        # gradients that are still multiplied by the loss scale.
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in core.parameters() if p.requires_grad],
                            grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in core.parameters() if p.requires_grad],
                            grad_clip)
                    optimizer.step()

                agg["loss"] += float(loss.detach())
                for k in ("best_loss", "fallback", "cand_valid", "cand_total",
                          "n", "unlikelihood", "best_valid"):
                    agg[k] += stats[k]
                for key in LOSS_TERMS:
                    agg[key] += stats[key]
                n_steps += 1
                global_step += 1

                if is_main:
                    pbar.set_postfix_str(
                        f"ep={epoch}/{num_epochs}  loss={float(loss.detach()):.4f}  "
                        f"best={stats['best_loss'] / max(stats['n'], 1):.3f}  "
                        f"fallback={stats['fallback'] / max(stats['n'], 1):.0%}  "
                        f"cand_valid={stats['cand_valid'] / max(stats['cand_total'], 1):.0%}",
                        refresh=True)
                pbar.update(1)

                # ONE read of the flag per batch, deliberately. Reading it
                # twice leaves a window: the signal lands after the save test
                # has seen False and before the break test sees True, so the
                # loop exits without writing the checkpoint the interrupt was
                # supposed to produce.
                stop_now = guard.requested
                if (stop_now or (ckpt_every and global_step % ckpt_every == 0)) \
                        and is_main:
                    provenance = lineage.save_state(
                        save_dir, STAGE, trainable=s10._trainable_state(core),
                        optimizer=optimizer.state_dict(), epoch=epoch,
                        batch_index=bi + 1, global_step=global_step,
                        history=history, agg=agg, n_steps=n_steps,
                        fingerprint=fp, rng=_rng_state(), profile=profile,
                        provenance=provenance)
                if stop_now:
                    stopped_early = True
                    break

            if stopped_early:
                break

            # ── epoch boundary ────────────────────────────────────────────
            n_mol = max(agg["n"], 1)
            row = {
                "epoch": epoch,
                "loss_mean": _all_reduce_mean(agg["loss"] / max(n_steps, 1),
                                              is_dist, device),
                "best_loss_mean": agg["best_loss"] / n_mol,
                "fallback_rate": agg["fallback"] / n_mol,
                "cand_valid_rate": agg["cand_valid"] / max(agg["cand_total"], 1),
                "best_valid_rate": agg["best_valid"] / n_mol,
                "unlikelihood_mean": agg["unlikelihood"] / n_mol,
                **{key: agg[key] / n_mol for key in LOSS_TERMS},
            }
            row["tox_alert_rate"] = s10.alert_rate_from_loss(row["tox_alert"])
            # Rank 0 only: the pass uses no collective, so the other
            # ranks simply proceed and meet it at the next epoch's
            # _shard_batches. Running it everywhere would score the same
            # molecules world_size times for one number.
            if is_main:
                row.update(split.validation_pass(
                    val_pairs,
                    lambda b: stage10_2_batch_loss(
                        b, tokenizer, model, device, pool,
                        variant=variant, k_cand=k_cand,
                        batched=batched, amp_dtype=amp_dtype),
                    model, LOSS_TERMS, batch_size=batch_size,
                    seed=seed))
            for k in s10.HISTORY_KEYS:
                if k in row:
                    history[k].append(row[k])
                elif len(history[k]) < len(history["epoch"]):
                    history[k].append(float("nan"))

            log(
                f"\n  Epoch {epoch} -- loss={row['loss_mean']:.4f}  "
                f"best_candidate_loss={row['best_loss_mean']:.3f}  "
                f"fallback={row['fallback_rate']:.1%}  "
                f"candidate_validity={row['cand_valid_rate']:.1%}  "
                f"target_validity={row['best_valid_rate']:.1%}\n"
                f"    loss split -- qed={row['qed']:.3f} sa={row['sa']:.3f} "
                f"novelty={row['novelty']:.3f} tox_alert={row['tox_alert']:.3f}"
                f"  (alert rate {row['tox_alert_rate']:.1%})"
                + (f"  unlikelihood={row['unlikelihood_mean']:.3f}"
                   if variant == "a" else "")
            )

            if is_main and row.get("val_n"):
                log(split.format_val_line(row))

            # MODEL SELECTION on the held-out fold, written the moment it
            # improves so an interrupted run leaves the BEST epoch on disk
            # rather than the last one.
            if is_main and split.SELECT_BEST_VAL and split.is_best_val(history):
                best_val = history["val_best_loss_mean"][-1]
                core.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                log(f"    new best held-out loss {best_val:.4f} "
                    f"-- model saved to {save_dir}")

            if is_main:
                if KEEP_EPOCH_CKPTS:
                    ep_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
                    # Stage 10's per-epoch format exactly, so its snapshots and
                    # these are interchangeable for inspection and for
                    # load_model_last_layers(checkpoint=...).
                    torch.save({"trainable": s10._trainable_state(core)}, ep_path)
                    log(f"  Epoch snapshot saved -> {ep_path}")
                # batch_index = 0 of the NEXT epoch: this epoch is finished.
                provenance = lineage.save_state(
                    save_dir, STAGE, trainable=s10._trainable_state(core),
                    optimizer=optimizer.state_dict(), epoch=epoch + 1,
                    batch_index=0, global_step=global_step, history=history,
                    agg=_fresh_agg(), n_steps=0, fingerprint=fp,
                    rng=_rng_state(), profile=profile, provenance=provenance)
            agg, n_steps = _fresh_agg(), 0

    pbar.close()

    if stopped_early:
        log(f"\n  Stopped at epoch {epoch}, step {global_step}. State saved to "
            f"{lineage.ckpt_path(save_dir, STAGE)}.\n"
            f"  Re-run the same command to continue from exactly here.")
        ddp_cleanup(is_dist)
        return history

    if is_main:
        # Skipped when an epoch was selected on the held-out fold: those
        # weights are already in save_dir and re-saving would overwrite the
        # SELECTED model with the LAST one, silently undoing the selection.
        if not (split.SELECT_BEST_VAL and best_val < float("inf")):
            core.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            log(f"\n  Final model saved to : {save_dir}")
        else:
            log(f"\n  Final model is the epoch with the lowest held-out "
                f"loss ({best_val:.4f}), already in : {save_dir}")
        if lineage.shared_enabled():
            log("  (shared mode: this is the LINEAGE model -- the endpoint of "
                "whichever stages trained it. Provenance:)")
            for line in lineage.provenance_table(provenance):
                log(line)
        # ".2a" not "2a": the variant is interpolated straight into the
        # filename and the title, so "2a" would read "stage102a".
        s10._plot_history(history, save_dir, f".2{variant}")
        s10._plot_tox_alert_rate(history, save_dir, f".2{variant}")
    ddp_cleanup(is_dist)
    return history


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(variant: str = None, max_pairs_per_source: int = None,
         sample_seed: int = None, workers: int = None, batched: bool = None,
         speed: str = None, fresh: bool = False,
         num_epochs: int = None) -> None:
    variant = (variant or VARIANT).lower()
    if speed is not None:
        global _SPEED_OVERRIDE
        _SPEED_OVERRIDE = resolve_speed(speed)
    save_dir = lineage.resolve_save_dir(STAGE, variant)
    is_main = int(os.environ.get("RANK", "0")) == 0

    if is_main:
        print("\n" + "=" * 62)
        print(f"STAGE 10.2{variant.upper()} -- SUPERVISED BEST-OF-K, "
              f"HARDWARE-TUNED")
        print("=" * 62)
        print(f"""
  Stage 10's objective, unchanged. What differs is execution -- and here,
  unlike Stage 10.1, some of it changes which candidates are drawn:

    - ONE padded forward and ONE multinomial per batch instead of one per
      molecule                                     [STAGE10_2_BATCHED_FORWARD]
    - the {K_CANDIDATES} x batch RDKit measurements per step across a
      process pool                                 [STAGE10_2_SCORING_WORKERS]
    - bf16/fp16 autocast, TF32, length bucketing   [STAGE10_2_SPEED]
    - multi-GPU under torchrun, batch split so the step count holds
    - resumable at every {CKPT_EVERY or 'epoch'} optimizer step(s), with
      Adam's moments and the RNG streams

  Set STAGE10_2_SPEED = "off" for a run that is bit-identical to Stage 10,
  or "safe" for one that adds only the pool. At "fast", compare against
  Stage 10 distributionally -- never by diffing outputs.

  loss(valid)   = {W_QED}*(1-QED) + {W_SA}*(SA-1)/9 + {W_NOVELTY}*similarity + {W_TOX_ALERT}*alert
  loss(invalid) = {LOSS_INVALID}   (worst valid = {S_WORST_VALID}, plus w_valid = {W_VALID})
  variant       = {variant}  ({"CE + unlikelihood on invalid" if variant == "a" else "CE only"})
  mask percent  = {config.MASK_PERCENT}%   (shared with Stage 9, 10 and 10.1)""")
        for line in lineage.describe_mode(STAGE, variant):
            print(line)
        print()

    pairs = collect_all_training_pairs(
        max_pairs=getattr(config, "STAGE10_MAX_TRAINING_PAIRS", None),
        max_per_parent=getattr(config, "STAGE10_MAX_PAIRS_PER_PARENT", None),
    )
    if not pairs:
        print("  No training pairs found. Run stage1a/stage1b first.")
        sys.exit(1)

    history = run_stage10_2_training(
        pairs=pairs, save_dir=save_dir, variant=variant,
        workers=workers, batched=batched, fresh=fresh,
        **({} if num_epochs is None else {"num_epochs": num_epochs}))

    if not is_main:
        return

    print("\n" + "=" * 62)
    print(f"  Stage 10.2{variant} training complete.")
    if history.get("epoch"):
        print(f"  Final training loss        : {history['loss_mean'][-1]:.4f}")
        print(f"  Final candidate validity   : {history['cand_valid_rate'][-1]:.1%}")
        print(f"  Final parent-fallback rate : {history['fallback_rate'][-1]:.1%}")
    print(f"  Model : {save_dir}")
    print("=" * 62)

    if os.path.isfile(os.path.join(save_dir, "config.json")):
        s10.run_stage10_eval(save_dir, f".2{variant}",
                             max_pairs_per_source, sample_seed)
    else:
        print("  Run stopped before the final model was written -- skipping the "
              "evaluation pass. Re-run to finish training first.")


def _parse_args(argv: list) -> tuple:
    """
    --variant a|b, --limit N ("none"/"all"/0 = uncapped), --seed N,
    --epochs N (overrides config.STAGE10_NUM_EPOCHS),
    --workers N, --speed off|safe|fast|auto, --no-batch, --fresh,
    --shared / --separate.
    """
    variant = speed = None
    limit = seed = workers = epochs = None
    for flag in ("--variant", "--limit", "--seed", "--workers", "--speed", "--epochs"):
        if flag not in argv:
            continue
        idx = argv.index(flag)
        if idx + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        raw = argv[idx + 1]
        if flag == "--variant":
            variant = raw.lower()
        elif flag == "--limit":
            limit = 0 if raw.lower() in ("none", "all", "0") else int(raw)
        elif flag == "--workers":
            workers = int(raw)
        elif flag == "--speed":
            speed = raw.lower()
        elif flag == "--epochs":
            epochs = int(raw)
        else:
            seed = int(raw)
    batched = False if "--no-batch" in argv else None
    if "--shared" in argv:
        lineage.set_shared_override(True)
    elif "--separate" in argv:
        lineage.set_shared_override(False)
    return (variant, limit, seed, workers, speed, batched,
            "--fresh" in argv, epochs)


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    """
    Pins the claims this file makes, in order of what would hurt most if wrong.

      0. The knob precedence rule, including that TF32 = False actually reaches
         torch.backends rather than merely being printed.
      1. At speed "off" and "safe" the loss is BIT-IDENTICAL to Stage 10's.
         This is what makes the fast path's cost measurable rather than
         assumed.
      2. The batched path is a valid Stage 10 step: same shape of statistics,
         a real gradient, and -- under the deterministic top_k = 1 setting,
         where sampling has no randomness left to reorder -- the SAME loss as
         Stage 10 to floating-point tolerance. That is what pins the
         gather/repeat/index bookkeeping end to end.
      3. Batching does not leak across molecules: padding never contributes,
         and a batch of one molecule matches that molecule scored alone.
      4. The lineage: shared mode puts all three stages in one directory,
         provenance records who trained what, a mixed-execution resume warns,
         and an interrupted run resumes to the same place.
    """
    import tempfile

    warnings.filterwarnings("ignore")
    from stage10_self_test import _build_pairs

    print("\n" + "=" * 62)
    print("  STAGE 10.2 SELF-TEST")
    print("=" * 62)

    global _SPEED_OVERRIDE
    saved_speed = _SPEED_OVERRIDE

    # ── 0. knob precedence, and TF32 actually landing ─────────────────────
    try:
        _SPEED_OVERRIDE = "off"
        off = describe_hardware_settings()
        assert off["batched_forward"] is False, "'off' must not batch"
        assert off["scoring_workers"] == 0, "'off' must score serially"
        assert off["amp"] is False and off["tf32"] is False, "'off' must be fp32"
        assert off["length_bucketing"] is False, "'off' must not reorder batches"
        assert off["stage10_identical"] is True, "'off' must claim Stage 10 parity"

        _SPEED_OVERRIDE = "safe"
        safe = describe_hardware_settings()
        assert safe["batched_forward"] is False and safe["amp"] is False
        assert safe["length_bucketing"] is False
        assert safe["stage10_identical"] is True, "'safe' must keep parity"

        _SPEED_OVERRIDE = "fast"
        fast = describe_hardware_settings()
        assert fast["tf32"] is True and fast["length_bucketing"] is True

        # An explicit config knob must beat the preset.
        config.STAGE10_2_LENGTH_BUCKETING = False
        try:
            assert hw_setting("length_bucketing") is False, "knob must beat preset"
        finally:
            config.STAGE10_2_LENGTH_BUCKETING = None
        assert hw_setting("length_bucketing") is True, "preset must return"

        try:
            resolve_speed("turbo")
            raise AssertionError("an unknown preset must raise, not default")
        except ValueError:
            pass
        print("  [0] knob precedence (arg > STAGE10_2_<KNOB> > STAGE10_2_SPEED), "
              "presets, unknown-preset raise   OK")

        # This stage's namespace must be SEALED against Stage 9.1's. The pool
        # is shared machinery whose keyword defaults resolve through Stage
        # 9.1's config, so every one of them is passed explicitly here -- and
        # maxtasksperchild is the one that would leak, because its "unset"
        # value is None, which ScoringPool reads as "not specified".
        saved_9_1 = getattr(config, "STAGE9_1_POOL_MAXTASKSPERCHILD", None)
        saved_9_1_blas = getattr(config, "STAGE9_1_WORKER_BLAS_THREADS", None)
        try:
            config.STAGE9_1_POOL_MAXTASKSPERCHILD = 7
            config.STAGE9_1_WORKER_BLAS_THREADS = 5
            pool = ScoringPool(
                0,
                start_method     = resolve_pool_start_method(),
                chunk_factor     = int(hw_setting("pool_chunk_factor")),
                maxtasksperchild = resolve_pool_maxtasks(),
                blas_threads     = resolve_worker_blas_threads(),
            )
            assert not pool.maxtasksperchild, (
                f"Stage 9.1's POOL_MAXTASKSPERCHILD leaked into this stage's "
                f"pool ({pool.maxtasksperchild!r}); STAGE10_2_* must be sealed")
            assert pool.blas_threads == 1, (
                f"Stage 9.1's WORKER_BLAS_THREADS leaked in "
                f"({pool.blas_threads!r})")
        finally:
            config.STAGE9_1_POOL_MAXTASKSPERCHILD = saved_9_1
            config.STAGE9_1_WORKER_BLAS_THREADS = saved_9_1_blas
        print("  [0c] the shared ScoringPool takes its shape from STAGE10_2_*, "
              "never from STAGE9_1_*   OK")

        # TF32 = False must WRITE the backend flag, not merely print "off".
        _SPEED_OVERRIDE = "off"
        info = apply_hardware_settings()
        assert info["tf32"] is False
        if torch.cuda.is_available():
            assert torch.backends.cuda.matmul.allow_tf32 is False, (
                "STAGE10_2_TF32=False must reach torch.backends, not just the banner")
        _SPEED_OVERRIDE = "fast"
        info = apply_hardware_settings()
        if torch.cuda.is_available() and get_profile().supports_tf32:
            assert torch.backends.cuda.matmul.allow_tf32 is True
        print("  [0b] STAGE10_2_TF32 reaches torch.backends in both directions   OK")

        # ── 1. parity: 'off'/'safe' == Stage 10, to the bit ───────────────
        tokenizer, model, device = s10.load_model_last_layers()
        model.train()
        PAIRS = _build_pairs(tokenizer)

        torch.manual_seed(1234)
        ref_loss, ref_stats = s10.stage10_batch_loss(
            PAIRS, tokenizer, model, device, variant="a", k_cand=6)

        for n_workers, label in ((0, "serial"), (2, "2-worker pool")):
            torch.manual_seed(1234)
            with ScoringPool(n_workers) as pool:
                got_loss, got_stats = stage10_2_batch_loss(
                    PAIRS, tokenizer, model, device, pool, variant="a",
                    k_cand=6, batched=False, amp_dtype=None)
            assert float(got_loss) == float(ref_loss), (
                f"{label}: {float(got_loss)!r} != Stage 10's {float(ref_loss)!r}")
            assert got_stats == ref_stats, f"{label}: stats differ"
            assert got_loss.requires_grad, f"{label}: loss must carry a gradient"
        print(f"  [1] unbatched loss == Stage 10's to zero tolerance "
              f"({float(ref_loss):.6f}), serial and pooled   OK")

        # ── 2. the batched path, pinned where nothing random is left ──────
        # TWO sources of randomness separate the two paths, and both have to be
        # removed before an equality assertion means anything:
        #
        #   top_k = 1   leaves torch.multinomial no choice, so the RNG-ORDER
        #               difference batching introduces has nothing to change.
        #   model.eval()  disables DROPOUT. This is the subtler one and it is
        #               worth naming: dropout draws once per FORWARD, so B
        #               separate forwards consume B masks while one padded
        #               forward consumes one. Under model.train() the two paths
        #               therefore differ by ~1.2 in the mask logits -- not a
        #               bug, and not float associativity either, but a real and
        #               unavoidable consequence of batching that no seed can
        #               align. It is orthogonal to the index bookkeeping this
        #               test exists to pin, so it is switched off here and
        #               stated in the module docstring instead.
        #
        # With both gone, any remaining difference is padding leaking into real
        # positions or a gather/repeat/scatter index being wrong -- which is
        # exactly what should fail loudly.
        model.eval()
        try:
            with ScoringPool(0) as pool:
                torch.manual_seed(99)
                det_ref, det_ref_stats = s10.stage10_batch_loss(
                    PAIRS, tokenizer, model, device, variant="a", k_cand=3,
                    top_k=1)
                torch.manual_seed(99)
                det_got, det_got_stats = stage10_2_batch_loss(
                    PAIRS, tokenizer, model, device, pool, variant="a",
                    k_cand=3, top_k=1, batched=True, amp_dtype=None)
            d = abs(float(det_got) - float(det_ref))
            assert d < 5e-5, (
                f"with dropout off and top_k=1 the batched path must reproduce "
                f"Stage 10's loss; differ by {d:.3e} "
                f"({float(det_got)} vs {float(det_ref)})")
            for k in ("n", "cand_total", "cand_valid", "fallback", "skipped"):
                assert det_got_stats[k] == det_ref_stats[k], (
                    f"at top_k=1, stat {k!r} differs: "
                    f"{det_got_stats[k]} vs {det_ref_stats[k]}")
            print(f"  [2] batched forward reproduces Stage 10 with dropout off "
                  f"at top_k=1 (|d| = {d:.2e}, identical statistics)   OK")

            # ── 3. no cross-molecule leakage from padding ─────────────────
            # PAIRS have different token lengths, so a batch of them pads. If
            # the mask index or the [B, K, L] scatter were wrong, or if padding
            # reached a real position, a molecule scored inside a padded batch
            # would differ from the same molecule scored alone.
            lengths = {len(tokenizer(m)["input_ids"]) for m, _ in PAIRS}
            assert len(lengths) > 1, (
                "this check is vacuous unless the test pairs differ in length")
            with ScoringPool(0) as pool:
                torch.manual_seed(5)
                solo, _ = stage10_2_batch_loss(
                    [PAIRS[0]], tokenizer, model, device, pool, variant="b",
                    k_cand=3, top_k=1, batched=True, amp_dtype=None)
                torch.manual_seed(5)
                in_batch, _ = stage10_2_batch_loss(
                    PAIRS, tokenizer, model, device, pool, variant="b",
                    k_cand=3, top_k=1, batched=True, amp_dtype=None)
                torch.manual_seed(5)
                solo_ref, _ = s10.stage10_batch_loss(
                    [PAIRS[0]], tokenizer, model, device, variant="b",
                    k_cand=3, top_k=1)
            assert abs(float(solo) - float(solo_ref)) < 5e-5, (
                f"a single-molecule batch must match Stage 10: "
                f"{float(solo)} vs {float(solo_ref)}")
            print(f"  [3] a padded batch spans {sorted(lengths)} tokens; a "
                  f"single-molecule batch still matches Stage 10 exactly   OK")
        finally:
            model.train()

        # A real gradient, reaching only the unfrozen tensors. Back in
        # train() mode, i.e. the configuration training actually runs in.
        with ScoringPool(0) as pool:
            torch.manual_seed(7)
            bl, bstats = stage10_2_batch_loss(
                PAIRS, tokenizer, model, device, pool, variant="a", k_cand=4,
                batched=True, amp_dtype=None)
        assert bl.requires_grad and bstats["n"] > 0
        model.zero_grad()
        bl.backward()
        got_grad = [n for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None
                    and float(p.grad.abs().sum()) > 0]
        frozen_grad = [n for n, p in model.named_parameters()
                       if not p.requires_grad and p.grad is not None]
        assert got_grad, "the batched loss must produce gradients"
        assert not frozen_grad, "frozen parameters must not receive gradients"
        model.zero_grad()
        print(f"  [2b] batched loss back-propagates into {len(got_grad)} "
              f"unfrozen tensors and no frozen ones   OK")

        # ── 4. the lineage ────────────────────────────────────────────────
        import stage10_lineage as lin

        lin.set_shared_override(True)
        dirs = {s: lin.resolve_save_dir(s, "a") for s in lin.STAGES}
        assert len(set(dirs.values())) == 1, (
            f"shared mode must put all three stages in ONE directory, got {dirs}")
        lin.set_shared_override(False)
        dirs = {s: lin.resolve_save_dir(s, "a") for s in lin.STAGES}
        assert len(set(dirs.values())) == 3, (
            f"separate mode must give three distinct directories, got {dirs}")
        # 10a and 10b never merge, in either mode.
        lin.set_shared_override(True)
        assert lin.resolve_save_dir("stage10", "a") != \
               lin.resolve_save_dir("stage10", "b"), \
            "variants are different objectives and must never share a directory"
        print("  [4] shared -> one directory, separate -> three, variants "
              "never merge   OK")

        # Provenance: segments coalesce while the execution holds, and a new
        # one opens the moment it changes.
        p10   = lin.execution_profile("stage10", amp=None, batched=False)
        p10_2 = lin.execution_profile("stage10_2", speed="fast", amp="bf16",
                                      batched=True)
        prov = lin.append_segment([], p10, epoch=1, global_step=10)
        prov = lin.append_segment(prov, p10, epoch=2, global_step=20)
        assert len(prov) == 1 and prov[0]["to_epoch"] == 2, (
            "an unchanged execution must extend its segment, not open a new one")
        prov = lin.append_segment(prov, p10_2, epoch=3, global_step=30)
        assert len(prov) == 2 and prov[1]["stage"] == "stage10_2"
        assert lin.execution_differs(prov[0], p10_2), "the change must be detected"
        banner = lin.lineage_banner({"provenance": prov[:1]}, p10_2)
        assert banner and "MIXED-EXECUTION" in banner, (
            "a cross-stage resume must warn")
        assert lin.lineage_banner({"provenance": prov[1:]}, p10_2) is None, (
            "an unchanged execution must NOT warn")
        print(f"  [4b] provenance coalesces to {len(prov)} segments, detects the "
              f"execution change and warns exactly once   OK")

        # Round-trip through the shared checkpoint, written by one stage and
        # read by another -- the property the whole mode rests on.
        with tempfile.TemporaryDirectory() as td:
            lin.set_shared_override(True)
            state = {"lm_head.weight": torch.zeros(3, 4)}
            prov2 = lin.save_state(
                td, "stage10", trainable=state, optimizer={"state": {}},
                epoch=3, batch_index=17, global_step=812,
                history={"epoch": [1, 2]}, fingerprint={"n_pairs": 5},
                profile=p10, provenance=[])
            back = lin.load_state(td, "stage10_2")
            assert back is not None, (
                "a checkpoint written by Stage 10 must be findable by Stage 10.2")
            assert back["epoch"] == 3 and back["batch_index"] == 17
            assert back["global_step"] == 812
            assert back["written_by"] == "stage10"
            assert back["provenance"] == prov2
            assert os.path.isfile(os.path.join(td, lin.LINEAGE_META_NAME)), (
                "the plain-text sidecar must be written next to the weights")
            # And in separate mode the same directory yields nothing for a
            # stage that did not write there.
            lin.set_shared_override(False)
            assert lin.load_state(td, "stage10_2") is None, (
                "separate mode must NOT pick up another stage's checkpoint")
            print("  [4c] a Stage 10 checkpoint resumes under Stage 10.2 in "
                  "shared mode and is invisible in separate mode   OK")
        lin.set_shared_override(None)

        # ── 5. interrupt mid-epoch and resume to the same place ───────────
        with tempfile.TemporaryDirectory() as td:
            _SPEED_OVERRIDE = "off"          # parity path: fully deterministic
            pairs = list(PAIRS) * 3          # 12 pairs -> 6 batches of 2
            kw = dict(pairs=pairs, save_dir=td, variant="b", num_epochs=1,
                      batch_size=2, k_cand=4, workers=0, ckpt_every=1,
                      auto_resume=True, batched=False)

            STOP_AFTER = 3
            real_guard = s10_1._InterruptGuard

            class _StopAfterNBatches(real_guard):
                reads = 0

                @property
                def requested(self):             # type: ignore[override]
                    type(self).reads += 1
                    return type(self).reads >= STOP_AFTER

                @requested.setter
                def requested(self, value):
                    pass

            globals()["_InterruptGuard"] = _StopAfterNBatches
            try:
                run_stage10_2_training(**kw)
            finally:
                globals()["_InterruptGuard"] = real_guard

            mid = lineage.load_state(td, STAGE)
            assert mid is not None, "an interrupted run must leave a checkpoint"
            assert mid["epoch"] == 1 and mid["batch_index"] == STOP_AFTER
            assert 0 < mid["batch_index"] < 6, "the stop must be genuinely mid-epoch"
            assert mid["optimizer"]["state"], "Adam moments must be checkpointed"
            assert mid["rng"]["torch"] is not None, "RNG state must be checkpointed"
            assert mid["provenance"], "every save must record provenance"
            assert mid["provenance"][-1]["stage"] == STAGE
            assert not os.path.isfile(os.path.join(td, "config.json")), (
                "an interrupted run must NOT write the final model")
            print(f"  [5] interrupt left a checkpoint at epoch 1 batch "
                  f"{mid['batch_index']}/6 with Adam moments, RNG and "
                  f"provenance   OK")

            hist = run_stage10_2_training(**kw)
            assert hist["epoch"] == [1], f"resume must finish epoch 1, got {hist}"
            assert os.path.isfile(os.path.join(td, "config.json")), (
                "a finished run must save the model for the eval pass")
            done = lineage.load_state(td, STAGE)
            assert done["epoch"] == 2 and done["batch_index"] == 0
            assert done["global_step"] == 6, (
                f"6 batches must have run across the interrupt, got "
                f"{done['global_step']}")
            print(f"  [6] resumed from batch {STOP_AFTER}, ran the remaining "
                  f"{6 - STOP_AFTER}, wrote the final model   OK")

    finally:
        _SPEED_OVERRIDE = saved_speed
        lineage.set_shared_override(None)

    print("\nStage 10.2 self-test passed.")


if __name__ == "__main__":
    # Required before any pool is created: under "spawn" the children re-import
    # this module, and without the guard they would re-run training recursively.
    mp.freeze_support()
    if "--hardware" in sys.argv:
        _args = _parse_args(sys.argv)
        if _args[4]:
            _SPEED_OVERRIDE = resolve_speed(_args[4])
        print(get_profile().describe())
        print("\n  STAGE 10.2 RESOLVED SETTINGS  "
              "(config.STAGE10_2_* over config.STAGE10_2_SPEED over hardware)")
        for _k, _v in describe_hardware_settings().items():
            print(f"    {_k:<20}: {_v}")
        print("=" * 68)
        sys.exit(0)
    if "--test" in sys.argv:
        _run_self_test()
    else:
        (_v, _limit, _seed, _workers, _speed, _batched, _fresh,
         _epochs) = _parse_args(sys.argv)
        main(variant=_v, max_pairs_per_source=_limit, sample_seed=_seed,
             workers=_workers, batched=_batched, speed=_speed, fresh=_fresh,
             num_epochs=_epochs)

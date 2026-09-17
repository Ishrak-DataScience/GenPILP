# -*- coding: utf-8 -*-
"""
stage10_4_tox21_aware_training.py
==================================
Stage 10.4 -- Stage 10.2's execution, with a LEARNED toxicity term added to
the objective.

Stage 10.2 changed how Stage 10 runs and left the objective alone; this file
does the opposite. Every speedup, every resolver, the pool, AMP, DDP, length
bucketing, the step-granular checkpoint -- all of it is imported from Stage
10.2 unchanged and reads the SAME config.STAGE10_2_* knobs. The one difference
is a fifth term in the composite loss:

    Stage 10.2  loss(valid) = w_qed*(1-QED) + w_sa*(SA-1)/9
                            + w_novelty*similarity + w_tox_alert*alert
    Stage 10.4  loss(valid) = ... all of the above ...
                            + w_tox21*(1 - Tox21_clean_probability)

Holding execution constant is deliberate: it is what makes a 10.2 vs 10.4
comparison a measurement of the OBJECTIVE rather than a confound of two
things changing at once. There is no STAGE10_4_SPEED, and adding one would
break that.

TWO TOXICITY TERMS, AND WHY BOTH
---------------------------------
The family already had a toxicity term -- w_tox_alert, the PAINS/Brenk
structural-alert filter. It is a RULE: a hand-curated list of substructures
that medicinal chemists learned to distrust. It is free, always available, and
binary. It also cannot say anything about a molecule whose substructures all
happen to be on nobody's list.

w_tox21 is a MEASUREMENT: a classifier fit to 7,831 compounds actually run
through 12 nuclear-receptor and stress-response assays. It is continuous, it
generalises to substructures no list contains, and it is exactly as good as
the checkpoint behind it -- which is why stage10_3_tox21_train.py reports a
scaffold-split AUC per assay and this file prints it at startup.

They disagree usefully. Keeping both, rather than replacing the alert term, is
what lets the training curves show whether the two toxicity notions move
together on generated molecules or pull apart.

WHERE THE TOX21 FORWARD RUNS, AND WHY IT IS NOT IN THE POOL
------------------------------------------------------------
Stage 9.1 settled this and stage9_1_scoring_worker.py documents it: the RDKit
terms are pure functions of two strings and run in worker processes; the Tox21
classifier is a torch model, so loading it once per worker would multiply its
memory by the pool size and have W processes contend for one GPU. It is scored
instead as ONE batched forward on the parent and merged with the workers'
measurements afterwards.

This file follows that exactly, and adds two reductions on top, because Stage
10 scores K candidates per molecule where Stage 9 scored one -- at B=16, K=16
that is 256 classifier rows per optimizer step rather than 16:

    valid-only   An invalid candidate is charged LOSS_INVALID and its tox21
                 term is never read. Scoring it would be a forward pass whose
                 result is discarded. ~70% of candidates are invalid at
                 MASK_PERCENT=15, so this is the larger of the two savings.
    dedup        Identical SMILES get identical scores -- the classifier is
                 frozen and in eval() -- so the batch is deduplicated before
                 the forward and the results are scattered back. This is
                 EXACT, not an approximation. K draws from a top-20
                 distribution repeat often.

Both are on by default and both are switchable, because "the optimisation is
exact" is a claim the self-test has to be able to check against the
unoptimised path.

THE CHECKPOINT IS A HARD REQUIREMENT, AND FAILS LOUDLY
-------------------------------------------------------
stage9.score_tox21 returns 0.0 -- "maximally toxic" -- when no classifier is
configured. That is the right fail-safe for a SCORE, where the term is being
maximised and an unverifiable molecule should not be rewarded. As a LOSS term
it is a trap: 0.0 clean means every candidate is charged the full w_tox21,
which is a CONSTANT, and a constant added to all K candidates cannot change
which one wins best-of-K. The run would train exactly like Stage 10.2 while
plotting a toxicity curve pinned at 100%, and nothing would look wrong.

So this file refuses to start with w_tox21 > 0 and no checkpoint. Run
stage10_3_tox21_train.py first, or set STAGE10_4_W_TOX21 = 0 and accept that
you are running Stage 10.2 under a different name.

SHARED-OUTPUT MODE IS NOT AVAILABLE HERE
-----------------------------------------
config.STAGE10_SHARED_OUTPUT lets a run begun under Stage 10 be finished under
10.1 or 10.2, because those three optimise an IDENTICAL objective and differ
only in execution. Stage 10.4 does not: its loss has a fifth term and its
LOSS_INVALID is 2.20 rather than 2.00. Continuing a 10.2 checkpoint here would
be training one model on two different objectives and calling it one run, so
lineage registers stage10_4 as private-only and this file ignores --shared.

Usage
-----
  python stage10_3_tox21_train.py                first: build the classifier
  python stage10_4_tox21_aware_training.py
  python stage10_4_tox21_aware_training.py --test
  ... --hardware        the machine profile and every RESOLVED knob, then exit
  ... --w-tox21 0.3     override config.STAGE10_4_W_TOX21 for this run
  ... --speed off       Stage 10's execution (still with the tox21 term)
  ... --variant b       train 10.4b instead of config.STAGE10_VARIANT
  ... --epochs N        override config.STAGE10_NUM_EPOCHS for this run
  ... --no-batch        per-molecule forward and sampler
  ... --fresh           ignore any checkpoint and start over
  ... --limit none      uncapped final property pass

  torchrun --nproc_per_node=4 stage10_4_tox21_aware_training.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import config
import stage10_data_split as split
import stage10_lineage as lineage
import stage10_vanila_backpropagation_training as s10
import stage10_1_parallel_RDKit_scoring_resumable_training as s10_1
import stage10_2_hardware_tuned_batched_AMP_DDP_training as s10_2
from hardware_autotune import get_profile
from stage9_masked_property_finetune import (
    _TOX21_AVAILABLE,
    collect_all_training_pairs,
    score_tox21_batch,
)

# Execution comes from Stage 10.2 WHOLESALE -- see the module docstring. These
# are listed by name rather than star-imported so a rename there fails here at
# import time instead of silently falling back to a default.
from stage10_2_hardware_tuned_batched_AMP_DDP_training import (
    _all_reduce_mean,
    _autocast,
    _device_type,
    _InterruptGuard,
    _restore_rng,
    _rng_state,
    _shard_batches,
    amp_label,
    apply_hardware_settings,
    batches_for_epoch,
    ddp_cleanup,
    ddp_setup,
    describe_hardware_settings,
    hw_setting,
    resolve_amp,
    resolve_batched,
    resolve_ddp_backend,
    resolve_grad_scaler,
    resolve_max_seq_tokens,
    resolve_pool_maxtasks,
    resolve_pool_start_method,
    resolve_speed,
    resolve_worker_blas_threads,
    resolve_workers,
    ScoringPool,
)

try:
    from tqdm import tqdm
except ImportError:                                   # pragma: no cover
    from tqdm_compat import tqdm  # type: ignore[misc]


STAGE = "stage10_4"

# ── knobs re-exported from Stage 10, so there is exactly one definition ──────
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

# ── the one new knob, and what it does to the loss scale ─────────────────────
# ADDED ON TOP of Stage 10's four weights rather than renormalising them. The
# consequence is deliberate: every existing term keeps its exact meaning, so
# 10.4's qed / sa / novelty / tox_alert curves are directly comparable with
# 10.2's, and only the TOTALS shift. Renormalising would have kept the total on
# 10.2's scale at the cost of making every per-term curve incomparable.
W_TOX21 = getattr(config, "STAGE10_4_W_TOX21", 0.20)

# The most a VALID molecule can lose, extended. This is not bookkeeping: an
# invalid molecule scores S + W_VALID, and if S did not grow with the new term
# then a valid-but-toxic molecule could out-lose an invalid one, inverting the
# "validity matters most" ordering the whole selection rule rests on.
S_WORST_VALID = s10.S_WORST_VALID + W_TOX21
LOSS_INVALID  = S_WORST_VALID + W_VALID

LOSS_TERMS = tuple(s10.LOSS_TERMS) + ("tox21",)

# Stage 10's series plus the two this stage adds. tox21_clean_rate is a pure
# function of tox21, exactly as tox_alert_rate is of tox_alert, so a resumed
# history missing it can be backfilled rather than left with a NaN gap.
HISTORY_KEYS = (tuple(s10.HISTORY_KEYS)
                + ("tox21", "tox21_clean_rate")
                # ...and their held-out counterparts. s10.HISTORY_KEYS already
                # carries val_ series for ITS four loss terms; the fifth is
                # this stage's, so its val_ key has to be added here or
                # validation_pass would return a number the history loop
                # silently drops.
                + ("val_tox21", "val_tox21_clean_rate"))

# Execution knobs deliberately come from Stage 10.2's namespace; see the
# docstring. Only the things Stage 10.2 could not already express get a
# STAGE10_4_* name.
TOX21_SUBBATCH   = getattr(config, "STAGE10_4_TOX21_SUBBATCH", 0)
TOX21_VALID_ONLY = getattr(config, "STAGE10_4_TOX21_VALID_ONLY", True)
TOX21_DEDUP      = getattr(config, "STAGE10_4_TOX21_DEDUP", True)

SEED = (getattr(config, "STAGE10_4_SEED", None)
        or getattr(config, "STAGE10_2_SEED", None)
        or getattr(config, "STAGE10_1_SEED", 42))
CKPT_EVERY = getattr(config, "STAGE10_4_CHECKPOINT_EVERY_STEPS", None)
if CKPT_EVERY is None:
    CKPT_EVERY = s10_2.CKPT_EVERY
KEEP_EPOCH_CKPTS = getattr(config, "STAGE10_4_KEEP_EPOCH_CHECKPOINTS", None)
if KEEP_EPOCH_CKPTS is None:
    KEEP_EPOCH_CKPTS = s10_2.KEEP_EPOCH_CKPTS


def describe_stage10_4_settings(speed: str = None) -> Dict[str, object]:
    """
    Stage 10.2's resolved-knob table, with the four rows that are about STAGE
    IDENTITY rather than about execution corrected for this stage.

    describe_hardware_settings closes over Stage 10.2's STAGE constant and its
    STAGE10_2_BATCH_SIZE, which is right for the execution knobs -- those ARE
    Stage 10.2's here, deliberately -- and wrong for these four. Left alone,
    --hardware would tell you Stage 10.4 writes to Stage 10.2's directory,
    which is the sort of thing someone reads once, believes, and then spends an
    hour looking for missing checkpoints because of.

    stage10_identical is dropped rather than corrected: it asks "is this run
    bit-identical to Stage 10", and for Stage 10.4 the answer is no under every
    possible setting, because the objective has a term Stage 10 does not.
    Reporting False would imply some other setting could make it True.
    """
    out = dict(describe_hardware_settings(speed))
    out.pop("stage10_identical", None)
    out["batch_size"] = (getattr(config, "STAGE10_4_BATCH_SIZE", None)
                         or getattr(config, "STAGE10_2_BATCH_SIZE", None)
                         or s10.BATCH_SIZE)
    out["shared_output"] = lineage.shared_enabled(stage=STAGE)
    out["output_dir"] = lineage.resolve_save_dir(STAGE, VARIANT)
    return out


def set_speed_override(name: Optional[str]) -> None:
    """
    Point Stage 10.2's process-wide speed override, since every resolver this
    file uses is Stage 10.2's and reads that global. Writing it here rather
    than shadowing it locally is what keeps --speed honest: the flag has to
    reach the code that actually resolves the knobs.
    """
    s10_2._SPEED_OVERRIDE = None if name is None else resolve_speed(name)


# ════════════════════════════════════════════════════════════════════════════
#  THE COMPOSITE LOSS  --  Stage 10.2's, plus one term
# ════════════════════════════════════════════════════════════════════════════

def compose_stage10_4_loss(
    measured: Dict[str, Optional[float]],
) -> Tuple[float, Dict[str, float]]:
    """
    Stage 10's composite loss with the Tox21 classifier term added.

    Built BY CALLING Stage 10.1's compose_stage10_loss rather than by restating
    its four terms, so the shared part cannot drift: if Stage 10's weights or
    formula change, they change here too, and _run_self_test asserts that
    subtracting this function's tox21 term reproduces Stage 10.2's loss to the
    bit on valid molecules, invalid strings and unmeasurable descriptors.

    `measured` is a compute_property_components dict with "tox21" filled in by
    attach_tox21 -- 1.0 = the classifier is confident the molecule is clean,
    0.0 = confident it is toxic. The loss charges (1 - clean), so lower is
    better, matching every other term here.

    THE INVALID BRANCH IS NOT A SPECIAL CASE, and that is the point. Stage
    10.1 already returns its LOSS_INVALID (S + w_valid) for an unparseable
    molecule; adding the full w_tox21 to it yields exactly (S + w_tox21) +
    w_valid, which is this module's LOSS_INVALID. The margin between the worst
    valid molecule and any invalid one stays exactly w_valid, unchanged from
    Stage 10.

    A missing measurement is charged its FULL weight -- the same pessimistic
    reading Stage 10 uses everywhere: anything unverifiable is treated as bad.
    That covers a classifier that failed on one string and, deliberately, a
    classifier that is not configured at all (score_tox21 fails safe to 0.0
    clean). Which is why _require_tox21_checkpoint refuses to let a run start
    in that state -- a term that is constant across candidates is invisible to
    best-of-K selection.
    """
    total, comps = s10_1.compose_stage10_loss(measured)

    if not comps["valid"]:
        comps["tox21"] = W_TOX21
        return float(total) + W_TOX21, comps

    clean = measured.get("tox21")
    l_tox21 = (W_TOX21 * (1.0 - float(clean)) if clean is not None else W_TOX21)
    comps["tox21"] = l_tox21
    return float(total) + l_tox21, comps


def clean_from_loss(tox21_term: float) -> float:
    """
    Turn the tox21 LOSS term back into the mean CLEAN PROBABILITY the
    classifier assigned to the molecules this stage trained toward.

    The term is w_tox21 * (1 - clean) averaged over the epoch's molecules, so
    clean = 1 - term/w_tox21 recovers it exactly. Molecules that were invalid,
    or that the classifier could not score, were charged the full weight
    upstream and so count as clean = 0 here -- unverified is treated as toxic,
    the same pessimistic reading the loss itself uses. That is stated on the
    figure, because an early-epoch curve sitting near zero otherwise reads as
    a broken classifier rather than as the ~70% invalid-candidate rate it
    mostly is.
    """
    if W_TOX21 <= 0:
        return 0.0
    return 1.0 - (float(tox21_term) / W_TOX21)


def new_history() -> Dict[str, list]:
    """An empty history in the shape this stage writes."""
    return {k: [] for k in HISTORY_KEYS}


def ensure_history_keys(history: Dict[str, list]) -> Dict[str, list]:
    """
    Widen a RESUMED history to this stage's key set, in place.

    Stage 10's version handles its own twelve series, including deriving
    tox_alert_rate; this adds the two Stage 10.4 series on top. tox21_clean_
    rate is derived from tox21 where the checkpoint has it -- exactly, since
    one is a pure function of the other -- and NaN otherwise, so matplotlib
    leaves a visible gap rather than drawing an invented value.
    """
    s10.ensure_history_keys(history)
    n = len(history.get("epoch") or [])
    # Every key this stage adds on top of Stage 10's -- the tox21 term, its
    # derived clean rate, and both of their held-out counterparts. Listing them
    # by difference rather than by hand is what stops a key added to
    # HISTORY_KEYS from being silently absent on resume, which is exactly how
    # the val_tox21 series was missed the first time.
    for key in (k for k in HISTORY_KEYS if k not in s10.HISTORY_KEYS):
        series = history.setdefault(key, [])
        if len(series) >= n:
            continue
        derived_from = {"tox21_clean_rate": "tox21",
                        "val_tox21_clean_rate": "val_tox21"}.get(key)
        if derived_from and len(history.get(derived_from) or []) >= n:
            history[key] = [clean_from_loss(v)
                            for v in history[derived_from][:n]]
        else:
            # Prepend: a short series is short because it started late, so its
            # values belong at the RECENT end of the epoch axis.
            series[:0] = [float("nan")] * (n - len(series))
    return history


# ════════════════════════════════════════════════════════════════════════════
#  THE TOX21 FORWARD  --  one batched pass on the parent, never in the pool
# ════════════════════════════════════════════════════════════════════════════

def attach_tox21(
    measured:   List[Dict[str, Optional[float]]],
    generated:  Sequence[str],
    subbatch:   int  = None,
    valid_only: bool = None,
    dedup:      bool = None,
) -> List[Dict[str, Optional[float]]]:
    """
    Fill measured[i]["tox21"] for a whole batch of candidates, IN PLACE.

    This is Stage 9.1's score_batch merge, with the two reductions the module
    docstring describes. Both are exact:

      valid_only  An invalid candidate takes compose_stage10_4_loss's invalid
                  branch, which charges the full w_tox21 without reading
                  measured["tox21"]. Scoring it would be a forward pass whose
                  result is provably discarded. Their entries are left None.
      dedup       The classifier is frozen and in eval(), so two identical
                  SMILES produce identical logits. Scoring the unique strings
                  and scattering the results back is the same arithmetic with
                  fewer rows.

    subbatch splits the forward when one [N, 256] allocation through a second
    transformer is what exhausts the card -- the same knob, and the same
    exactness argument, as config.STAGE9_1_TOX21_SUBBATCH: no row's logits
    depend on any other row's.

    Returns `measured` for convenience. A no-op when the classifier is
    unconfigured or w_tox21 is 0, leaving every entry None, which the loss
    then charges at full weight.
    """
    subbatch   = TOX21_SUBBATCH   if subbatch   is None else subbatch
    valid_only = TOX21_VALID_ONLY if valid_only is None else bool(valid_only)
    dedup      = TOX21_DEDUP      if dedup      is None else bool(dedup)

    if not _TOX21_AVAILABLE or W_TOX21 <= 0 or not measured:
        return measured

    wanted = [i for i, m in enumerate(measured)
              if (m.get("valid") or not valid_only)]
    if not wanted:
        return measured

    if dedup:
        first_at: Dict[str, int] = {}
        unique: List[str] = []
        for i in wanted:
            s = generated[i]
            if s not in first_at:
                first_at[s] = len(unique)
                unique.append(s)
        slots = [first_at[generated[i]] for i in wanted]
    else:
        unique = [generated[i] for i in wanted]
        slots = list(range(len(unique)))

    step = int(subbatch) or len(unique) or 1
    clean: List[float] = []
    for i in range(0, len(unique), step):
        clean.extend(score_tox21_batch(unique[i:i + step]))

    for i, slot in zip(wanted, slots):
        measured[i]["tox21"] = clean[slot]
    return measured


def _require_tox21_checkpoint(log=None) -> None:
    """
    Refuse to train with a weighted term the classifier cannot supply.

    The failure this prevents is silent, not loud: with no checkpoint,
    score_tox21 returns 0.0 for every molecule, so every candidate is charged
    the identical full w_tox21. A constant is invisible to an argmin over K
    candidates, so the run would train EXACTLY like Stage 10.2 while drawing a
    toxicity curve flat at zero clean probability. Nothing about that looks
    like a bug from the outside, which is precisely why it has to be an error.
    """
    if W_TOX21 <= 0 or _TOX21_AVAILABLE:
        return
    raise SystemExit(
        "\n  Stage 10.4 needs a Tox21 checkpoint.\n\n"
        f"  config.STAGE9_TOX21_MODEL_DIR is "
        f"{getattr(config, 'STAGE9_TOX21_MODEL_DIR', '')!r} and "
        f"STAGE10_4_W_TOX21 is {W_TOX21}.\n"
        "  Without the classifier, score_tox21 returns 0.0 for every molecule,\n"
        "  so the term is the same constant for all K candidates and cannot\n"
        "  change which one wins best-of-K. The run would train identically to\n"
        "  Stage 10.2 while plotting a toxicity curve that never moves.\n\n"
        "  Fix it either way:\n"
        "    python stage10_3_tox21_train.py      # builds the checkpoint,\n"
        "                                         # then set STAGE9_TOX21_MODEL_DIR\n"
        "    STAGE10_4_W_TOX21 = 0                # run Stage 10.2's objective\n")


# ════════════════════════════════════════════════════════════════════════════
#  ONE TRAINING STEP
# ════════════════════════════════════════════════════════════════════════════

def stage10_4_batch_loss(
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
    Stage 10.2's batch loss with the Tox21 forward spliced between scoring and
    selection. Returns the same (loss, stats) pair, with one extra stats key.

    Four phases now, not three:

        phase 1  sample K candidates for every molecule                 (GPU)
                   batched=True   one padded [B, L] forward, one multinomial
                                  over every mask in the batch
                   batched=False  one forward and one sampler call per
                                  molecule -- Stage 10's RNG consumption order
        phase 2  measure all B x K candidates with RDKit                (pool)
        phase 2b ONE classifier forward over the valid, deduplicated
                 candidates, merged into the same dicts             (this GPU)
        phase 3  compose, select best per molecule, build the CE       (here)

    UNLIKE Stage 10.2, THE batched=False PATH IS NOT DELEGATED. Stage 10.2 can
    hand its parity path to Stage 10.1 because their objectives are identical;
    this one's is not, so both paths are built here and share phases 2, 2b and
    3 exactly. What `batched` selects is only how phase 1 draws -- which is the
    honest scope of the flag anyway.

    The two paths produce the same per-molecule record shape, so phase 3 is
    written once. In the batched path a molecule's rows are contiguous and
    ascending in b_idx because nonzero() scans row-major, which is the same
    mask order parent_target_ids returns its tokens in.

    Bit-identity, restated for this stage: at batched=False with the pool and
    fp32, Stage 10.4 is bit-identical to a hypothetical "Stage 10 plus tox21",
    not to Stage 10.2 -- the objective differs by construction. What the
    self-test CAN pin, and does, is that at W_TOX21 = 0 this function
    reproduces Stage 10.2's loss exactly.
    """
    variant             = (variant or VARIANT).lower()
    k_cand              = K_CANDIDATES        if k_cand is None else k_cand
    top_k               = TOP_K               if top_k is None else top_k
    temperature         = TEMPERATURE         if temperature is None else temperature
    fallback_weight     = FALLBACK_WEIGHT     if fallback_weight is None else fallback_weight
    unlikelihood_weight = UNLIKELIHOOD_WEIGHT if unlikelihood_weight is None else unlikelihood_weight
    batched             = resolve_batched() if batched is None else bool(batched)

    stats: Dict[str, float] = {
        "n": 0.0, "fallback": 0.0, "cand_valid": 0.0, "cand_total": 0.0,
        "best_loss": 0.0, "unlikelihood": 0.0, "skipped": 0.0,
        "best_valid": 0.0, "tox21_scored": 0.0,
    }
    for key in LOSS_TERMS:
        stats[key] = 0.0

    # ── phase 1 ───────────────────────────────────────────────────────────
    pending: List[dict] = []
    flat: List[Tuple[str, str]] = []
    gen_only: List[str] = []

    if batched:
        if tokenizer.padding_side != "right":
            raise ValueError(
                f"tokenizer.padding_side must be 'right' for the batched "
                f"forward, got {tokenizer.padding_side!r}.")

        # STAGE10_2_MAX_SEQ_TOKENS is a CEILING, never a raise: a tokenizer
        # reporting a shorter window than the knob still wins, because
        # exceeding the model's positional range is a crash, not a slow run.
        cap = resolve_max_seq_tokens(max_tokens)
        tok_max = getattr(tokenizer, "model_max_length", cap)
        if tok_max is None or tok_max > 1024:
            tok_max = cap
        max_len = min(int(tok_max), cap)

        masked_list = [m for m, _ in batch]
        parents     = [p for _, p in batch]
        enc = tokenizer(masked_list, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(device)
        ids       = enc["input_ids"]                    # [B, L]
        attn_mask = enc["attention_mask"]               # [B, L]
        B, L = ids.shape

        with _autocast(device, amp_dtype):
            out = model(input_ids=ids, attention_mask=attn_mask)

        # Padding is never selected here: pad tokens are not mask tokens.
        b_idx, l_idx = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            stats["skipped"] += float(B)
            return torch.zeros((), device=device, requires_grad=True), stats

        # .float() before any softmax: under fp16 autocast a log_softmax over
        # a 767-wide vocabulary loses meaningful precision, and these values
        # feed both the selection and the gradient. The cast touches only the
        # masked rows, not the whole [B, L, V].
        sel = out.logits[b_idx, l_idx].float()          # [M, V], carries grad
        M, V = sel.shape

        scaled = (sel / max(temperature, 1e-8)).detach()
        probs  = F.softmax(scaled, dim=-1)
        kk     = min(top_k, V)
        top    = torch.topk(probs, k=kk, dim=-1)
        renorm = top.values / top.values.sum(dim=-1, keepdim=True)
        picks  = torch.multinomial(renorm, num_samples=k_cand, replacement=True)
        cand_ids = top.indices.gather(1, picks)         # [M, K]

        # filled[b, j, :] is molecule b with candidate j written into its
        # masks. The advanced indices sit on dims 0 and 2 with a slice between
        # them, so the broadcast result is [M, K] -- cand_ids' shape exactly.
        filled = ids.unsqueeze(1).repeat(1, k_cand, 1)  # [B, K, L]
        filled[b_idx, :, l_idx] = cand_ids
        decoded = tokenizer.batch_decode(filled.view(B * k_cand, L),
                                         skip_special_tokens=True)

        rows_by_mol: List[List[int]] = [[] for _ in range(B)]
        for row, b in enumerate(b_idx.tolist()):
            rows_by_mol[b].append(row)

        for b in range(B):
            rows = rows_by_mol[b]
            if not rows:
                stats["skipped"] += 1
                continue
            row_idx = torch.tensor(rows, device=device, dtype=torch.long)
            smiles = [decoded[b * k_cand + j].replace(" ", "")
                      for j in range(k_cand)]
            pending.append({
                "masked": masked_list[b], "parent": parents[b],
                "at_masks": sel[row_idx],                   # [n_masks, V]
                "cands":    cand_ids[row_idx].t().contiguous(),  # [K, n_masks]
            })
            gen_only.extend(smiles)
            flat.extend((s, parents[b]) for s in smiles)
    else:
        # The per-molecule path: Stage 10's forward and sampler, molecule by
        # molecule, consuming the RNG in Stage 10's order.
        for masked_smi, parent_smi in batch:
            enc = tokenizer(masked_smi, return_tensors="pt", truncation=True,
                            max_length=MAX_MODEL_TOKENS).to(device)
            ids = enc["input_ids"][0]
            mask_pos = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if mask_pos.numel() == 0:
                stats["skipped"] += 1
                continue

            logits = model(**enc).logits[0]             # [L, V], carries grad
            at_masks = logits[mask_pos]                 # [n_masks, V]
            cands = s10._sample_candidates(at_masks, k_cand, top_k, temperature)

            smiles = []
            for row in cands:
                one = ids.clone()
                one[mask_pos] = row
                smiles.append(tokenizer.decode(
                    one, skip_special_tokens=True).replace(" ", ""))

            pending.append({"masked": masked_smi, "parent": parent_smi,
                            "at_masks": at_masks, "cands": cands})
            gen_only.extend(smiles)
            flat.extend((s, parent_smi) for s in smiles)

    if not pending:
        return torch.zeros((), device=device, requires_grad=True), stats

    # ── phase 2: RDKit, in the pool ───────────────────────────────────────
    measured = pool.measure(flat)

    # ── phase 2b: the classifier, here, once ──────────────────────────────
    attach_tox21(measured, gen_only)
    stats["tox21_scored"] = float(
        sum(1 for m in measured if m.get("tox21") is not None))

    # ── phase 3: compose, select, differentiate ───────────────────────────
    total = torch.zeros((), device=device)
    n_weighted = 0.0
    cursor = 0

    for item in pending:
        cands    = item["cands"]
        at_masks = item["at_masks"]
        n_masks  = at_masks.shape[0]

        best_loss, best_ids, best_comps = None, None, None
        invalid_rows: List[torch.Tensor] = []
        for j in range(len(cands)):
            loss_c, comps = compose_stage10_4_loss(measured[cursor])
            cursor += 1
            stats["cand_total"] += 1
            if comps["valid"]:
                stats["cand_valid"] += 1
            else:
                invalid_rows.append(cands[j])
            if best_loss is None or loss_c < best_loss:
                best_loss, best_ids, best_comps = loss_c, cands[j], comps

        used_fallback = (best_comps is None) or (not best_comps["valid"])
        if used_fallback:
            tgt = s10.parent_target_ids(item["masked"], item["parent"], tokenizer)
            if tgt is None or len(tgt) != n_masks:
                stats["skipped"] += 1
                continue
            best_ids = torch.tensor(tgt, device=device, dtype=torch.long)
            weight = fallback_weight
            stats["fallback"] += 1
        else:
            weight = 1.0

        log_probs = F.log_softmax(at_masks, dim=-1)
        ce = F.nll_loss(log_probs, best_ids, reduction="mean")
        mol_loss = weight * ce

        if variant == "a" and invalid_rows and unlikelihood_weight > 0:
            bad = torch.stack(invalid_rows)             # [n_bad, n_masks]
            keep = bad != best_ids.unsqueeze(0)
            if keep.any():
                p = log_probs.exp()
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
#  CHECKPOINTING
# ════════════════════════════════════════════════════════════════════════════

def _fingerprint(variant: str, batch_size, k_cand: int, num_epochs: int,
                 n_pairs: int, seed: int, bucketing: bool) -> Dict[str, object]:
    """
    Stage 10.2's fingerprint plus the two fields that define THIS stage's
    objective.

    w_tox21 and the classifier directory are in here, and neither is in Stage
    10.2's, because they are the only settings that can change what the loss
    MEANS without changing a single thing about which molecules land in which
    batch. Resuming across a change to either is continuing one run under two
    objectives -- the resume path warns loudly rather than silently averaging
    them into one set of curves.
    """
    fp = s10_2._fingerprint(variant, batch_size, k_cand, num_epochs, n_pairs,
                            seed, bucketing)
    fp["w_tox21"] = float(W_TOX21)
    fp["tox21_model"] = str(getattr(config, "STAGE9_TOX21_MODEL_DIR", "") or "")
    return fp


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_stage10_4_training(
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
    Stage 10.2's loop, running Stage 10.4's objective.

    The loop is restated here rather than parameterised out of Stage 10.2, for
    the same reason Stage 10.2 restated Stage 10.1's: each stage owning its own
    loop is what lets one be edited without the others silently changing. Every
    HELPER is imported.
    """
    if speed is not None:
        set_speed_override(speed)

    variant = (variant or VARIANT).lower()
    if variant not in ("a", "b"):
        raise ValueError(f"variant must be 'a' or 'b', got {variant!r}")
    save_dir = save_dir or lineage.resolve_save_dir(STAGE, variant)

    _require_tox21_checkpoint()

    # Stage 10.2's knobs, deliberately -- holding execution identical is what
    # makes 10.2 vs 10.4 a measurement of the objective. See the docstring.
    workers = (resolve_workers(getattr(config, "STAGE10_2_SCORING_WORKERS", None))
               if workers is None else resolve_workers(workers))
    batched = resolve_batched() if batched is None else bool(batched)
    bucketing = bool(hw_setting("length_bucketing"))
    if ckpt_every is None:
        ckpt_every = CKPT_EVERY
    if batch_size is None:
        batch_size = (getattr(config, "STAGE10_4_BATCH_SIZE", None)
                      or getattr(config, "STAGE10_2_BATCH_SIZE", None)
                      or s10.BATCH_SIZE)

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
        batch_size = hw.recommended_batch_size(
            seq_len=128, vocab=767, n_pairs=len(pairs), n_epochs=num_epochs,
            min_total_steps=int(hw_setting("hw_min_total_steps")),
            max_batch=int(hw_setting("hw_max_batch")),
            headroom=float(hw_setting("hw_mem_headroom")),
        )
        log(f"  Batch size {batch_size} chosen by hardware_autotune "
            f"(>= {int(hw_setting('hw_min_total_steps'))} total optimizer "
            f"steps, cap {int(hw_setting('hw_max_batch'))}).")

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

    history: Dict[str, list] = new_history()

    def _fresh_agg() -> Dict[str, float]:
        a = {k: 0.0 for k in ("loss", "best_loss", "fallback", "cand_valid",
                              "cand_total", "n", "unlikelihood", "best_valid")}
        for key in LOSS_TERMS:
            a[key] = 0.0
        return a

    fp = _fingerprint(variant, global_batch, k_cand, num_epochs, len(pairs),
                      seed, bucketing)

    # ── model first: the AMP dtype is part of this run's identity ─────────
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

    def _want_resume() -> bool:
        if auto_resume is not None:
            return bool(auto_resume)
        if is_dist:
            log("  Distributed run -- resuming automatically (pass --fresh to "
                "start over instead).")
            return True
        return s10_1._ask_resume(save_dir, ckpt)

    if ckpt and _want_resume():
        banner = lineage.lineage_banner(ckpt, profile)
        if banner:
            log(banner)
        old_fp = ckpt.get("fingerprint", {})
        batching_same = all(old_fp.get(k) == fp[k]
                            for k in ("n_pairs", "batch_size", "seed", "bucketing"))
        start_epoch = int(ckpt["epoch"])
        global_step = int(ckpt["global_step"])
        history     = ensure_history_keys(ckpt["history"])
        provenance  = ckpt.get("provenance") or []
        resume_state = ckpt
        if batching_same:
            start_batch = int(ckpt["batch_index"])
            agg     = ckpt.get("agg") or _fresh_agg()
            n_steps = int(ckpt.get("n_steps", 0))
        else:
            changed = [k for k in ("n_pairs", "batch_size", "seed", "bucketing")
                       if old_fp.get(k) != fp[k]]
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
        # Louder than the two above, because these change what the recorded
        # curves MEAN rather than how the run executes: the epochs already in
        # `history` were trained against a different objective, and plotting
        # them on one axis with what follows invites reading a step change in
        # the weight as a result.
        for k, what in (("w_tox21", "the toxicity weight"),
                        ("tox21_model", "the Tox21 checkpoint")):
            if old_fp.get(k) is not None and old_fp.get(k) != fp[k]:
                log(f"\n  WARNING: {what} changed since the checkpoint "
                    f"({old_fp.get(k)!r} -> {fp[k]!r}).\n"
                    f"  The {len(history.get('epoch') or [])} epoch(s) already "
                    f"in this history optimised a DIFFERENT objective. The "
                    f"curves\n  from here on are not comparable with them. "
                    f"--fresh starts a clean run.\n")
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
        torch.manual_seed(seed + rank)
    del resume_state

    scaler = None
    if resolve_grad_scaler(amp_dtype) and _device_type(device) == "cuda":
        init_scale = hw_setting("grad_scaler_init_scale")
        scaler = torch.amp.GradScaler(
            _device_type(device),
            **({} if init_scale is None else {"init_scale": float(init_scale)}),
        )

    tox_dir = getattr(config, "STAGE9_TOX21_MODEL_DIR", "") or "(none)"
    log(
        f"\n  Variant {variant} : "
        + ("CE toward best candidate + unlikelihood on invalid ones"
           if variant == "a" else "CE toward best candidate only")
        + f"\n  K candidates    : {k_cand}"
        + f"\n  batch size      : {batch_size}"
        + (f" per rank ({global_batch} global)" if is_dist and split_batch else "")
        + f"  ({n_batches_per_epoch} batches/epoch)"
        + f"\n  speed preset    : {resolve_speed(speed)}"
          f"   [config.STAGE10_2_SPEED -- execution is Stage 10.2's]"
        + f"\n  forward/sampler : "
        + ("BATCHED -- one padded forward and one multinomial per batch."
           if batched else "per-molecule -- Stage 10's RNG consumption order.")
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
        + f"\n  length bucketing: {'ON' if bucketing else 'OFF'}"
        + f"\n  checkpoint      : "
        + (f"every {ckpt_every} steps + every epoch" if ckpt_every
           else "every epoch only (mid-epoch saves disabled)")
        + f"\n  hardware        : {hw.environment}, {hw.cpu_count} CPU "
          f"({hw.cpu_source}), device {device}"
        + (f", {world_size} ranks" if is_dist else "")
        + f"\n  Tox21 classifier: {tox_dir}"
        + f"\n                    "
        + (f"valid-only={TOX21_VALID_ONLY}, dedup={TOX21_DEDUP}, "
           f"subbatch={TOX21_SUBBATCH or 'whole batch'}"
           if W_TOX21 > 0 else "term disabled (w_tox21 = 0)")
        + f"\n  loss (valid)    : {W_QED}*(1-QED) + {W_SA}*(SA-1)/9 "
          f"+ {W_NOVELTY}*similarity + {W_TOX_ALERT}*alert "
          f"+ {W_TOX21}*(1-tox21_clean)   -> at most {S_WORST_VALID}"
        + f"\n  loss (invalid)  : {LOSS_INVALID}  (= {S_WORST_VALID} + "
          f"w_valid {W_VALID})"
        + f"\n  fallback weight : {FALLBACK_WEIGHT}  (parent-reconstruction steps)"
    )

    total_remaining = ((num_epochs - start_epoch + 1) * n_batches_per_epoch
                       - start_batch)
    pbar = lineage.Progress(total=max(total_remaining, 0),
                            desc=f"Stage 10.4{variant} training",
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
            todo = list(range(first, len(batches)))
            todo = _shard_batches([[i] for i in todo], rank, world_size,
                                  is_dist, device)
            todo = [t[0] for t in todo]

            for bi in todo:
                batch = [pairs[i] for i in batches[bi]]
                optimizer.zero_grad()
                loss, stats = stage10_4_batch_loss(
                    batch, tokenizer, model, device, pool, variant=variant,
                    k_cand=k_cand, batched=batched, amp_dtype=amp_dtype)

                if stats["n"] == 0:
                    pbar.update(1)
                    continue

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        # unscale_ first, or the clip threshold is applied to
                        # gradients still multiplied by the loss scale.
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
                        f"ep={epoch}/{num_epochs}  "
                        f"loss={float(loss.detach()):.4f}  "
                        f"best={stats['best_loss'] / max(stats['n'], 1):.3f}  "
                        f"clean="
                        f"{clean_from_loss(stats['tox21'] / max(stats['n'], 1)):.2f}  "
                        f"fallback={stats['fallback'] / max(stats['n'], 1):.0%}",
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
            row["tox21_clean_rate"] = clean_from_loss(row["tox21"])
            # Rank 0 only: the pass uses no collective, so the other
            # ranks simply proceed and meet it at the next epoch's
            # _shard_batches. Running it everywhere would score the same
            # molecules world_size times for one number.
            if is_main:
                row.update(split.validation_pass(
                    val_pairs,
                    lambda b: stage10_4_batch_loss(
                        b, tokenizer, model, device, pool,
                        variant=variant, k_cand=k_cand,
                        batched=batched, amp_dtype=amp_dtype),
                    model, LOSS_TERMS, batch_size=batch_size,
                    seed=seed))
                # Same pure function of the tox21 term as on the training side,
                # so the two curves are the same quantity measured on two folds.
                if "val_tox21" in row:
                    row["val_tox21_clean_rate"] = clean_from_loss(row["val_tox21"])
            for k in HISTORY_KEYS:
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
                f" tox21={row['tox21']:.3f}\n"
                f"    toxicity   -- structural alerts "
                f"{row['tox_alert_rate']:.1%} of molecules, "
                f"Tox21 clean probability {row['tox21_clean_rate']:.3f}"
                + (f"\n    unlikelihood={row['unlikelihood_mean']:.3f}"
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
        # ".3a" not "3a": the variant is interpolated straight into the
        # filename and the title, so "3a" would read "stage103a".
        s10._plot_history(history, save_dir, f".4{variant}")
        s10._plot_tox_alert_rate(history, save_dir, f".4{variant}")
        s10._plot_validation_properties(history, save_dir, f".4{variant}")
        _plot_tox21_clean(history, save_dir, f".4{variant}")
        _plot_toxicity_comparison(history, save_dir, f".4{variant}")
        _plot_loss_split(history, save_dir, f".4{variant}")
    ddp_cleanup(is_dist)
    return history


# ════════════════════════════════════════════════════════════════════════════
#  FIGURES  --  the three this stage adds
# ════════════════════════════════════════════════════════════════════════════

def _plot_tox21_clean(history: Dict[str, list], save_dir: str,
                      variant: str) -> None:
    """
    Mean Tox21 clean probability against epoch -- the curve this whole stage
    exists to move.

    On its own axes, fixed to 0..1, for the reason _plot_tox_alert_rate is:
    this is read against an ABSOLUTE scale, and a shared-figure panel
    autoscales to whatever narrow band the run occupies, turning a two-point
    wobble into an apparent trend.

    The epoch-1 value is drawn as a reference line, because the question is
    never "is 0.62 good" -- it is "did it move, and by how much".
    """
    ep = history.get("epoch") or []
    if not ep:
        return
    clean = history.get("tox21_clean_rate") or []
    if len(clean) != len(ep):
        clean = [clean_from_loss(v) for v in (history.get("tox21") or [])]
    if len(clean) != len(ep):
        return
    finite = [(e, c) for e, c in zip(ep, clean) if c == c]
    if not finite:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([e for e, _ in finite], [c for _, c in finite],
            marker="o", color="#16a085", linewidth=2,
            label="selected (best-of-K) molecules")
    base_ep, base_c = finite[0]
    ax.axhline(base_c, color="#95a5a6", linestyle="--", linewidth=1,
               label=f"epoch {base_ep} baseline ({base_c:.3f})")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Tox21 clean probability  (1 - toxic)")
    ax.set_title(f"Stage 10{variant} -- learned toxicity of the training "
                 f"targets\nHIGHER IS CLEANER; w_tox21 = {W_TOX21}", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    ax.text(0.01, -0.16,
            "Invalid candidates and molecules the classifier could not score "
            "are counted as clean = 0\n(unverified is treated as toxic, the "
            "same pessimistic reading the loss uses), so early epochs sit low\n"
            "largely because ~70% of candidates are invalid at "
            f"MASK_PERCENT={getattr(config, 'MASK_PERCENT', 15)}.",
            transform=ax.transAxes, fontsize=7, color="#555555",
            va="top", ha="left")
    plt.tight_layout()
    out = os.path.join(save_dir, f"stage10{variant}_tox21_clean.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  Tox21 curve saved     : {out}")


def _plot_toxicity_comparison(history: Dict[str, list], save_dir: str,
                              variant: str) -> None:
    """
    The two toxicity notions on one axis: the PAINS/Brenk alert rate (a rule)
    against the Tox21 clean probability (a measurement).

    Both are plotted as "fraction of molecules judged clean" so that UP is
    better for both and the two curves are directly readable against each
    other. The alert series is inverted for this -- 1 - alert_rate -- and the
    axis label says so, because plotting one metric where lower is better
    beside one where higher is is the fastest way to misread a figure.

    This is the figure that answers the question adding the term raises: do
    the rule and the classifier agree about the molecules this stage selects,
    or does optimising one leave the other flat?
    """
    ep = history.get("epoch") or []
    if not ep:
        return
    alert = history.get("tox_alert_rate") or []
    clean = history.get("tox21_clean_rate") or []
    if len(alert) != len(ep) or len(clean) != len(ep):
        return
    alert_free = [1.0 - a for a in alert]

    pts = [(e, af, c) for e, af, c in zip(ep, alert_free, clean)
           if af == af and c == c]
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s",
            color="#c0392b", linewidth=2,
            label="alert-free (1 - PAINS/Brenk rate)  -- a RULE")
    ax.plot([p[0] for p in pts], [p[2] for p in pts], marker="o",
            color="#16a085", linewidth=2,
            label=f"Tox21 clean probability  -- a MEASUREMENT "
                  f"(w={W_TOX21})")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction judged clean  (up is better for both)")
    ax.set_title(f"Stage 10{variant} -- the two toxicity terms, compared",
                 fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out = os.path.join(save_dir, f"stage10{variant}_toxicity_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  Toxicity comparison   : {out}")


def _plot_loss_split(history: Dict[str, list], save_dir: str,
                     variant: str) -> None:
    """
    All five weighted terms stacked, so the composite loss can be read as what
    it is: a budget being reallocated.

    Stacked rather than overlaid because the terms SUM to the composite loss
    by construction (each is already weighted), so the stack height is the
    quantity being minimised and each band's thickness is that term's share of
    it. Overlaid lines would show the same numbers while hiding the one
    relationship that matters -- that a term can only shrink by another
    growing, unless the total falls.
    """
    ep = history.get("epoch") or []
    if not ep:
        return
    keys = [("qed", "QED", "#8c564b"), ("sa", "SA", "#e377c2"),
            ("novelty", "novelty (similarity)", "#9467bd"),
            ("tox_alert", "PAINS/Brenk alert", "#c0392b"),
            ("tox21", f"Tox21 (w={W_TOX21})", "#16a085")]
    series = []
    for k, _, _ in keys:
        s = history.get(k) or []
        if len(s) != len(ep):
            return
        series.append([0.0 if v != v else v for v in s])   # NaN -> 0 for stack

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(ep, *series, labels=[lbl for _, lbl, _ in keys],
                 colors=[c for _, _, c in keys], alpha=0.85)
    ax.axhline(S_WORST_VALID, color="#333333", linestyle="--", linewidth=1,
               label=f"worst possible valid molecule ({S_WORST_VALID})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted loss contribution (terms sum to the composite)")
    ax.set_title(f"Stage 10{variant} -- composite loss, by term\n"
                 f"invalid molecules score {LOSS_INVALID} "
                 f"(= {S_WORST_VALID} + w_valid {W_VALID})", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out = os.path.join(save_dir, f"stage10{variant}_loss_split.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  Loss split saved      : {out}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(variant: str = None, max_pairs_per_source: int = None,
         sample_seed: int = None, workers: int = None, batched: bool = None,
         speed: str = None, fresh: bool = False,
         num_epochs: int = None) -> None:
    variant = (variant or VARIANT).lower()
    if speed is not None:
        set_speed_override(speed)
    save_dir = lineage.resolve_save_dir(STAGE, variant)
    is_main = int(os.environ.get("RANK", "0")) == 0

    if is_main:
        print("\n" + "=" * 62)
        print(f"STAGE 10.4{variant.upper()} -- SUPERVISED BEST-OF-K, "
              f"TOX21-AWARE")
        print("=" * 62)
        print(f"""
  Stage 10.2's execution, unchanged and reading the same STAGE10_2_* knobs,
  with one term added to the objective. Holding execution fixed is what
  makes a 10.2 vs 10.4 comparison measure the OBJECTIVE.

    + w_tox21 * (1 - Tox21 clean probability)      [STAGE10_4_W_TOX21]
    - the classifier runs as ONE batched forward on this process, never in
      the RDKit pool (a torch model per worker would multiply memory by the
      pool size and contend for the GPU)
    - only VALID candidates are scored, and duplicates are scored once --
      both exact, both switchable      [STAGE10_4_TOX21_VALID_ONLY / _DEDUP]

  loss(valid)   = {W_QED}*(1-QED) + {W_SA}*(SA-1)/9 + {W_NOVELTY}*similarity
                  + {W_TOX_ALERT}*alert + {W_TOX21}*(1-tox21_clean)
  loss(invalid) = {LOSS_INVALID}   (worst valid = {S_WORST_VALID}, plus w_valid = {W_VALID})
  variant       = {variant}  ({"CE + unlikelihood on invalid" if variant == "a" else "CE only"})
  mask percent  = {config.MASK_PERCENT}%   (shared with Stage 9, 10, 10.1 and 10.2)
  Tox21 model   = {getattr(config, "STAGE9_TOX21_MODEL_DIR", "") or "NOT CONFIGURED"}""")
        _print_tox21_report()
        for line in lineage.describe_mode(STAGE, variant):
            print(line)
        print()

    _require_tox21_checkpoint()

    pairs = collect_all_training_pairs(
        max_pairs=getattr(config, "STAGE10_MAX_TRAINING_PAIRS", None),
        max_per_parent=getattr(config, "STAGE10_MAX_PAIRS_PER_PARENT", None),
    )
    if not pairs:
        print("  No training pairs found. Run stage1a/stage1b first.")
        sys.exit(1)

    history = run_stage10_4_training(
        pairs=pairs, save_dir=save_dir, variant=variant,
        workers=workers, batched=batched, fresh=fresh,
        **({} if num_epochs is None else {"num_epochs": num_epochs}))

    if not is_main:
        return

    print("\n" + "=" * 62)
    print(f"  Stage 10.4{variant} training complete.")
    if history.get("epoch"):
        print(f"  Final training loss        : {history['loss_mean'][-1]:.4f}")
        print(f"  Final candidate validity   : {history['cand_valid_rate'][-1]:.1%}")
        print(f"  Final parent-fallback rate : {history['fallback_rate'][-1]:.1%}")
        print(f"  Final alert rate           : {history['tox_alert_rate'][-1]:.1%}")
        print(f"  Final Tox21 clean prob.    : "
              f"{history['tox21_clean_rate'][-1]:.3f}")
        if len(history["tox21_clean_rate"]) > 1:
            first = history["tox21_clean_rate"][0]
            last  = history["tox21_clean_rate"][-1]
            print(f"    (epoch 1 -> {epoch_word(len(history['epoch']))}: "
                  f"{first:.3f} -> {last:.3f}, "
                  f"{'+' if last >= first else ''}{last - first:.3f})")
    print(f"  Model : {save_dir}")
    print("=" * 62)

    if os.path.isfile(os.path.join(save_dir, "config.json")):
        s10.run_stage10_eval(save_dir, f".4{variant}",
                             max_pairs_per_source, sample_seed)
    else:
        print("  Run stopped before the final model was written -- skipping "
              "the evaluation pass. Re-run to finish training first.")


def epoch_word(n: int) -> str:
    """'epoch 7' -- kept tiny and separate only so the summary line above
    stays readable."""
    return f"epoch {n}"


def _print_tox21_report() -> None:
    """
    Print the classifier's own scaffold-split AUC next to the weight it is
    about to be given.

    A w_tox21 of 0.20 against a 0.62-AUC classifier and against an 0.82-AUC one
    are very different experiments, and the report sits in the checkpoint
    directory where nobody would think to look during a training run.
    """
    import json

    d = getattr(config, "STAGE9_TOX21_MODEL_DIR", "") or ""
    path = os.path.join(d, "tox21_training_report.json") if d else ""
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path) as fh:
            rep = json.load(fh)
    except Exception:
        return
    print(f"  Tox21 quality : test mean AUC {rep.get('test_mean_auc', float('nan')):.3f} "
          f"({rep.get('split')} split, {rep.get('n_test')} held-out molecules)")
    aucs = rep.get("test_auc_per_task") or {}
    measured = {k: v for k, v in aucs.items() if v is not None}
    if measured:
        worst = min(measured.items(), key=lambda p: p[1])
        best  = max(measured.items(), key=lambda p: p[1])
        print(f"                  best {best[0]} {best[1]:.3f}, "
              f"worst {worst[0]} {worst[1]:.3f}")


def _parse_args(argv: list) -> tuple:
    """
    --variant a|b, --limit N ("none"/"all"/0 = uncapped), --seed N,
    --epochs N (overrides config.STAGE10_NUM_EPOCHS),
    --workers N, --speed off|safe|fast|auto, --w-tox21 F, --no-batch, --fresh.

    --shared is NOT accepted: Stage 10.4's objective differs from the rest of
    the family, so a shared checkpoint would mean two objectives in one run.
    """
    global W_TOX21, S_WORST_VALID, LOSS_INVALID

    variant = speed = None
    limit = seed = workers = epochs = None
    for flag in ("--variant", "--limit", "--seed", "--workers", "--speed",
                 "--w-tox21", "--epochs"):
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
        elif flag == "--w-tox21":
            # The three constants move together or the invalid-molecule margin
            # silently stops being w_valid.
            W_TOX21 = float(raw)
            S_WORST_VALID = s10.S_WORST_VALID + W_TOX21
            LOSS_INVALID = S_WORST_VALID + W_VALID
        else:
            seed = int(raw)
    batched = False if "--no-batch" in argv else None
    if "--shared" in argv:
        raise SystemExit(
            "  --shared is not available for Stage 10.4.\n"
            "  Shared mode exists so a run begun under Stage 10 can be "
            "finished under 10.1 or 10.2,\n  which is only sound because those "
            "three optimise an IDENTICAL objective. Stage 10.4\n  adds a fifth "
            "loss term, so continuing one of their checkpoints here would be "
            "training\n  one model on two objectives and reporting it as one "
            "run.")
    return (variant, limit, seed, workers, speed, batched,
            "--fresh" in argv, epochs)


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    """
    Pins the claims this file makes, in order of what would hurt most if wrong.

      1. The loss REDUCES to Stage 10.2's. Every term but tox21 is Stage
         10.1's function, untouched, and at w_tox21 = 0 the totals agree to
         the bit. This is what makes "10.4 = 10.2 + one term" a fact rather
         than a description.
      2. The invalid margin survives. An invalid molecule must still be worse
         than the worst valid one by exactly w_valid, after the new term grew
         the ceiling.
      3. clean_from_loss inverts the term exactly, so the plotted curve is the
         measurement and not a rescaling of it.
      4. attach_tox21's two reductions are EXACT: dedup and valid-only produce
         the same losses as scoring everything.
      5. A resumed history is widened, and the derived series is derived
         rather than NaN-filled.
      6. Missing measurements are charged full weight, in every direction.
    """
    global W_TOX21, S_WORST_VALID, LOSS_INVALID

    print("\nStage 10.4 self-test")
    print("=" * 62)

    def _measured(valid=True, qed=0.7, sa=3.0, nov=0.4, alert=0.0, tox21=None):
        return {"valid": valid, "qed": qed, "sa_raw": sa, "novelty": nov,
                "any_alert": alert, "tox21": tox21, "pains": None,
                "brenk": None, "n_alerts": None, "alert_free": None}

    saved_w = W_TOX21

    # ── 1. reduces to Stage 10.2 at w_tox21 = 0 ──────────────────────────
    try:
        W_TOX21 = 0.0
        S_WORST_VALID = s10.S_WORST_VALID
        LOSS_INVALID = S_WORST_VALID + W_VALID
        cases = [
            _measured(),
            _measured(alert=1.0),
            _measured(valid=False),
            _measured(qed=None, sa=None, nov=None, alert=None),
            _measured(nov=0.0),                       # a self-comparison
            _measured(tox21=0.9),                     # tox21 present, w = 0
        ]
        for m in cases:
            got, _ = compose_stage10_4_loss(dict(m))
            want, _ = s10_1.compose_stage10_loss(dict(m))
            assert got == want, (
                f"at w_tox21=0 Stage 10.4 must equal Stage 10.2 exactly; "
                f"got {got} want {want} on {m}")
        print("  [1] at w_tox21 = 0 the loss IS Stage 10.2's, to the bit   OK")
    finally:
        W_TOX21 = saved_w
        S_WORST_VALID = s10.S_WORST_VALID + W_TOX21
        LOSS_INVALID = S_WORST_VALID + W_VALID

    # ── 2. the shared terms are untouched, and only tox21 is added ───────
    for m in (_measured(tox21=1.0), _measured(tox21=0.0), _measured(tox21=0.5),
              _measured(alert=1.0, tox21=0.25)):
        got, comps = compose_stage10_4_loss(dict(m))
        base, base_comps = s10_1.compose_stage10_loss(dict(m))
        assert abs((got - comps["tox21"]) - base) < 1e-12, (
            f"subtracting the tox21 term must reproduce Stage 10.2's loss: "
            f"{got} - {comps['tox21']} != {base}")
        for k in ("qed", "sa", "novelty", "tox_alert"):
            assert comps[k] == base_comps[k], f"term {k} was modified"
    print("  [2] every other term is Stage 10.1's function, unmodified   OK")

    # ── 3. the invalid margin is exactly w_valid, after the ceiling grew ──
    worst_valid, comps = compose_stage10_4_loss(
        _measured(qed=0.0, sa=10.0, nov=0.0, alert=1.0, tox21=0.0))
    assert abs(worst_valid - S_WORST_VALID) < 1e-12, (
        f"the worst VALID molecule must lose exactly S_WORST_VALID "
        f"({S_WORST_VALID}), got {worst_valid}")
    invalid, _ = compose_stage10_4_loss(_measured(valid=False))
    assert abs(invalid - LOSS_INVALID) < 1e-12
    assert abs((invalid - worst_valid) - W_VALID) < 1e-12, (
        f"an invalid molecule must be worse than every valid one by exactly "
        f"w_valid ({W_VALID}); the margin is {invalid - worst_valid}")
    assert abs(S_WORST_VALID - (s10.S_WORST_VALID + W_TOX21)) < 1e-12
    print(f"  [3] worst valid {S_WORST_VALID}, invalid {LOSS_INVALID}, "
          f"margin exactly {W_VALID}   OK")

    # ── 4. an unmeasurable term is charged its FULL weight ───────────────
    no_tox, comps_no = compose_stage10_4_loss(_measured(tox21=None))
    yes_tox, comps_yes = compose_stage10_4_loss(_measured(tox21=1.0))
    assert comps_no["tox21"] == W_TOX21, (
        "a molecule the classifier could not score must be charged the full "
        "weight -- unverified is treated as toxic")
    assert comps_yes["tox21"] == 0.0, "a perfectly clean molecule pays nothing"
    assert no_tox > yes_tox
    print("  [4] unscored -> full weight, clean=1.0 -> zero              OK")

    # ── 5. clean_from_loss inverts the term exactly ──────────────────────
    for clean in (0.0, 0.25, 0.5, 0.9, 1.0):
        _, c = compose_stage10_4_loss(_measured(tox21=clean))
        back = clean_from_loss(c["tox21"])
        assert abs(back - clean) < 1e-12, (
            f"clean_from_loss({c['tox21']}) = {back}, expected {clean}")
    print("  [5] clean_from_loss inverts the loss term exactly           OK")

    # ── 6. attach_tox21: both reductions are exact ───────────────────────
    if _TOX21_AVAILABLE and W_TOX21 > 0:
        gen = ["CCO", "c1ccccc1", "CCO", "CCN", "c1ccccc1"]
        base = [dict(_measured()) for _ in gen]
        full  = attach_tox21([dict(m) for m in base], gen,
                             dedup=False, valid_only=False)
        dedup = attach_tox21([dict(m) for m in base], gen,
                             dedup=True, valid_only=False)
        for i, (a, b) in enumerate(zip(full, dedup)):
            assert a["tox21"] == b["tox21"], (
                f"dedup changed molecule {i}'s score: {a['tox21']} vs "
                f"{b['tox21']} -- it must be exact, the classifier is frozen")
        sub = attach_tox21([dict(m) for m in base], gen, subbatch=1,
                           dedup=False, valid_only=False)
        for a, b in zip(full, sub):
            assert abs(a["tox21"] - b["tox21"]) < 1e-9, \
                "sub-batching moved a score"
        mixed = [dict(_measured()), dict(_measured(valid=False))]
        out = attach_tox21(mixed, ["CCO", "not-a-molecule"], valid_only=True)
        assert out[0]["tox21"] is not None and out[1]["tox21"] is None, (
            "valid_only must leave invalid candidates unscored -- their loss "
            "never reads the value")
        # ... and leaving it unscored must not change what they lose.
        l_unscored, _ = compose_stage10_4_loss(out[1])
        assert abs(l_unscored - LOSS_INVALID) < 1e-12
        print("  [6] attach_tox21: dedup, sub-batching, valid-only exact    OK")
    else:
        print("  [6] attach_tox21 exactness SKIPPED "
              "(no Tox21 checkpoint configured)")

    # ── 7. history widening derives rather than NaN-fills ────────────────
    old = {"epoch": [1, 2], "loss_mean": [1.0, 0.9],
           "tox21": [W_TOX21 * 0.5, W_TOX21 * 0.25]}
    widened = ensure_history_keys(dict(old))
    assert len(widened["tox21_clean_rate"]) == 2
    assert abs(widened["tox21_clean_rate"][0] - 0.5) < 1e-12, (
        "tox21_clean_rate is a pure function of tox21 and must be DERIVED on "
        "resume, not NaN-filled")
    assert abs(widened["tox21_clean_rate"][1] - 0.75) < 1e-12
    for k in HISTORY_KEYS:
        assert len(widened[k]) == 2, f"series {k} was not widened to 2 epochs"
    print("  [7] a resumed history is widened, derived series derived    OK")

    # ── 8. the guard against a silently-inert term ───────────────────────
    if not _TOX21_AVAILABLE:
        try:
            _require_tox21_checkpoint()
        except SystemExit:
            print("  [8] refuses to train a weighted term with no classifier  OK")
        else:
            raise AssertionError(
                "with no checkpoint and w_tox21 > 0 the run MUST refuse to "
                "start -- the term would be constant and change nothing")
    else:
        _require_tox21_checkpoint()
        print("  [8] classifier configured; startup guard passes           OK")

    # ── 9. the fingerprint carries what defines the objective ────────────
    fp = _fingerprint("a", 16, 16, 3, 100, 42, True)
    assert fp["w_tox21"] == float(W_TOX21)
    assert "tox21_model" in fp
    for k in ("variant", "batch_size", "k_cand", "num_epochs", "n_pairs",
              "seed", "bucketing"):
        assert k in fp, f"Stage 10.2's fingerprint field {k} was dropped"
    print("  [9] fingerprint = Stage 10.2's + w_tox21 + the checkpoint   OK")

    # ── 10. lineage knows this stage, and shared mode is refused ─────────
    assert STAGE in lineage.STAGES, (
        f"{STAGE} is not registered in stage10_lineage.STAGES -- "
        f"resolve_save_dir and ckpt_path would raise")
    d = lineage.resolve_save_dir(STAGE, "a")
    assert "variant_a" in d.replace("\\", "/")
    # Shared mode must not reach this stage even when config asks for it: its
    # objective differs, so one checkpoint would hold two of them.
    saved_shared = lineage._SHARED_OVERRIDE
    try:
        lineage.set_shared_override(True)
        assert not lineage.shared_enabled(stage=STAGE), (
            "Stage 10.4 must never share a checkpoint, even with "
            "STAGE10_SHARED_OUTPUT / --shared on")
        assert lineage.shared_enabled(stage="stage10_2"), (
            "the override must still reach the stages that DO share")
        assert lineage.ckpt_path("d", STAGE).endswith("stage10_4_checkpoint.pt")
        # And the resolved-settings table must name THIS stage's directory,
        # not the one it borrows its execution knobs from.
        assert "stage10_4" in str(
            describe_stage10_4_settings()["output_dir"]).replace("\\", "/") \
            or lineage.base_dir(STAGE) in str(
                describe_stage10_4_settings()["output_dir"])
        assert describe_stage10_4_settings()["shared_output"] is False
    finally:
        lineage._SHARED_OVERRIDE = saved_shared
    try:
        _parse_args(["prog", "--shared"])
    except SystemExit:
        print("  [10] lineage registered; --shared refused                 OK")
    else:
        raise AssertionError("--shared must be refused for Stage 10.4")

    # ── 11. a checkpoint round-trips under this stage's name ─────────────
    # Registering a stage is three dict entries and a tuple; missing one shows
    # up only when a real run tries to save, hours in. This exercises the whole
    # path -- write, read back, and confirm the fields a resume depends on
    # survive -- against a temp directory, in about a millisecond.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        hist = new_history()
        hist["epoch"].append(1)
        hist["tox21"].append(W_TOX21 * 0.4)
        lineage.save_state(
            td, STAGE, trainable={"w": torch.zeros(2)}, optimizer=None,
            epoch=2, batch_index=0, global_step=7, history=hist,
            agg=None, n_steps=0, fingerprint=fp, rng=None,
            profile=lineage.execution_profile(STAGE, speed="fast"),
            provenance=[])
        assert os.path.isfile(lineage.ckpt_path(td, STAGE)), \
            "save_state wrote nothing under stage10_4's checkpoint name"
        back = lineage.load_state(td, STAGE)
        assert back["epoch"] == 2 and back["global_step"] == 7
        assert back["written_by"] == STAGE
        assert back["fingerprint"]["w_tox21"] == float(W_TOX21), (
            "w_tox21 must survive the round-trip -- the resume path compares "
            "it to detect an objective change")
        widened = ensure_history_keys(back["history"])
        assert abs(widened["tox21_clean_rate"][0] - 0.6) < 1e-12
    print("  [11] checkpoint save -> load round-trips as stage10_4      OK")

    print("\nStage 10.4 self-test passed.")
    if not _TOX21_AVAILABLE:
        print("  NOTE: no Tox21 checkpoint configured, so the exactness test "
              "for\n  attach_tox21 was skipped. Run stage10_3_tox21_train.py, "
              "point\n  config.STAGE9_TOX21_MODEL_DIR at it, and re-run "
              "--test to cover it.")


if __name__ == "__main__":
    # Required before any pool is created: under "spawn" the children re-import
    # this module, and without the guard they would re-run training recursively.
    mp.freeze_support()
    if "--hardware" in sys.argv:
        _args = _parse_args(sys.argv)
        if _args[4]:
            set_speed_override(_args[4])
        print(get_profile().describe())
        print("\n  STAGE 10.4 RESOLVED SETTINGS  "
              "(execution knobs are Stage 10.2's, by design)")
        for _k, _v in describe_stage10_4_settings().items():
            print(f"    {_k:<20}: {_v}")
        print(f"\n  STAGE 10.4 OBJECTIVE")
        print(f"    {'w_tox21':<20}: {W_TOX21}")
        print(f"    {'tox21 checkpoint':<20}: "
              f"{getattr(config, 'STAGE9_TOX21_MODEL_DIR', '') or 'NOT CONFIGURED'}")
        print(f"    {'classifier loaded':<20}: {_TOX21_AVAILABLE}")
        print(f"    {'worst valid loss':<20}: {S_WORST_VALID}")
        print(f"    {'invalid loss':<20}: {LOSS_INVALID}")
        print(f"    {'valid-only scoring':<20}: {TOX21_VALID_ONLY}")
        print(f"    {'dedup':<20}: {TOX21_DEDUP}")
        print(f"    {'tox21 subbatch':<20}: {TOX21_SUBBATCH or 'whole batch'}")
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

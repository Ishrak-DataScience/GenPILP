# -*- coding: utf-8 -*-
"""
stage10_1_parallel_RDKit_scoring_resumable_training.py
=======================================================
Stage 10.1 -- Stage 10's supervised best-of-K fine-tuning, re-executed.

SAME loss, SAME data, SAME batch size, SAME optimizer-step count. Only the
EXECUTION differs, so the weights this produces are directly comparable to
Stage 10's and the two can be run against each other as a pure speed check.

Everything scientific is imported from stage10_vanila_backpropagation_training
rather than re-implemented -- the composite loss weights, the model loader and
its freezing rule, the parent-target recovery, the candidate sampler, the
training-curve plot, the evaluation pass. Stage 10 is left untouched as the
reference implementation; if the objective changes there, it changes here.

The two things that are different
----------------------------------
1. RESUMABLE AT STEP GRANULARITY (this file's main purpose)

   Stage 10 checkpoints once per EPOCH: it writes epoch_NNN.pt plus a JSON
   holding last_epoch and history, and on restart continues from the next
   epoch. Three things are missing from that, and all three are fixed here.

   a. THE OPTIMIZER IS NOT SAVED. Stage 10 builds a fresh torch.optim.Adam
      after every resume, so exp_avg and exp_avg_sq -- the entire adaptive
      state that makes Adam Adam -- are discarded. Resume three times in a
      four-epoch run and you get three momentum restarts, each of which takes
      hundreds of steps to rebuild. Stage 9.1 already does this correctly
      (its optimizer.pt); Stage 10 has no equivalent.

   b. THERE IS NO MID-EPOCH STATE. An epoch is not a cheap unit here. Stage 10
      runs K = 16 RDKit measurements per molecule, so at batch 16 that is 256
      per batch, ~2.3 s of RDKit alone, ~2.5 h per epoch on the capped Stage 1b
      pool. Ctrl-C at 95% of an epoch throws away two and a half hours. This
      file checkpoints every config.STAGE10_1_CHECKPOINT_EVERY_STEPS steps and
      resumes at the exact batch it stopped on, so the worst case is N steps.

   c. NO RNG STATE IS KEPT, so a resumed run cannot reproduce the run it is
      continuing. torch's global stream drives torch.multinomial inside
      _sample_candidates, and Stage 10 reseeds its shuffle from
      random.Random(42 + start_epoch), which differs from what an
      uninterrupted run would have drawn. Both streams are checkpointed here.

   Ctrl-C is also caught: SIGINT sets a flag, the loop finishes the batch in
   flight, flushes a checkpoint and exits. A second Ctrl-C aborts immediately.
   Nothing is saved from inside the handler -- a signal arriving in the middle
   of a 93 MB torch.save is exactly how a checkpoint file gets truncated.

2. PARALLEL RDKit CANDIDATE SCORING (Stage 9.1's speedup 7b)

   RDKit dominates Stage 10 far more than it dominates Stage 9. Stage 9 scores
   one sampled molecule per training molecule; Stage 10 scores K = 16 of them.
   Those K x B candidates are independent, so they go to a process pool -- the
   same persistent spawn pool and the same worker module Stage 9.1 uses, which
   already calls compute_property_components with exactly the arguments Stage
   10's loss consumes (need_alert_count=False, need_tox21=False).

   To batch them into ONE pool.map, the per-molecule loop is split into three
   phases: sample and decode every candidate in the batch (GPU, sequential),
   measure all B x K of them at once (pool), then compose the loss, select the
   best per molecule and build the cross-entropy (parent process). No molecule
   is scored differently -- only later.

What is deliberately NOT ported from Stage 9.1
-----------------------------------------------
The batched GPU forward (7a), mixed precision, TF32, length bucketing and DDP
are all absent, and their absence is what buys the property below.

Stage 9.1 states that it will NOT reproduce Stage 9 token-for-token, because
batching changes the order in which the global RNG is consumed. That is an
acceptable trade there, where the score is a multiplier on a log-probability.
It is a worse trade in Stage 10, where the score SELECTS the training target:
perturb the logits (AMP) or the sampling order (batching) and a different
candidate wins best-of-K, so the model trains toward different tokens. The
difference stops being numerical and becomes a different target sequence.

So Stage 10.1 keeps the per-molecule forward and the per-molecule sampler
untouched. RDKit runs on the same strings in a different process, which cannot
change a float, and the epoch permutation is reconstructed to match Stage 10's
exactly (see _epoch_order). The result is that Stage 10.1 is BIT-IDENTICAL to
Stage 10 at the same seed -- _run_self_test asserts the batch losses match to
zero tolerance -- which is a much stronger guarantee than Stage 9.1 can offer,
and it means any difference you measure between the two runs is wall-clock.

Configuration
-------------
  config.STAGE10_1_DIR                      own output dir; never Stage 10's
  config.STAGE10_1_SCORING_WORKERS          "auto" | N | 0/1 for serial
  config.STAGE10_1_CHECKPOINT_EVERY_STEPS   0 disables mid-epoch saves
  config.STAGE10_1_KEEP_EPOCH_CHECKPOINTS   per-epoch snapshots on top
  config.STAGE10_1_AUTO_RESUME              True resumes without prompting
  config.STAGE10_1_BATCH_SIZE               None inherits STAGE10_BATCH_SIZE
  config.STAGE10_1_SEED                     42 reproduces Stage 10's shuffle

Every other knob -- the loss weights, K, top-k, temperature, the fallback
weight, the unlikelihood weight, which layers unfreeze, epochs, learning rate,
mask percent -- is Stage 10's, read from the same config entries.

Usage
-----
  python stage10_1_parallel_RDKit_scoring_resumable_training.py
  python stage10_1_parallel_RDKit_scoring_resumable_training.py --test
  ... --variant b        train 10b instead of config.STAGE10_VARIANT
  ... --epochs N        override config.STAGE10_NUM_EPOCHS for this run
  ... --workers 0        force serial scoring (compare against the pool)
  ... --fresh            ignore any checkpoint and start over
  ... --limit none       uncapped final property pass
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
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
from hardware_autotune import get_profile
from stage9_masked_property_finetune import collect_all_training_pairs

# ScoringPool and resolve_workers come from Stage 9.1 rather than being copied:
# the pool's lifecycle (persistent spawn context, worker initializer that
# pre-builds the PAINS/Brenk catalogs, terminate-on-exit) is execution
# machinery that both stages should share, and "auto" must mean the same number
# of workers in both or their timings are not comparable.
from stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning import (
    ScoringPool,
    resolve_workers,
)

try:
    from tqdm import tqdm
except ImportError:                                   # pragma: no cover
    # A silent stand-in with the same call surface, tqdm.write included;
    # the old inline stub was a bare function and had no .write.
    from tqdm_compat import tqdm  # type: ignore[misc]


# ── knobs ───────────────────────────────────────────────────────────────────
# Re-exported from Stage 10 so there is exactly one definition of each. Listed
# explicitly rather than star-imported so that a rename in Stage 10 fails here
# loudly at import time instead of silently falling back to a default.
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

BATCH_SIZE = getattr(config, "STAGE10_1_BATCH_SIZE", None) or s10.BATCH_SIZE
SEED       = getattr(config, "STAGE10_1_SEED", 42)
CKPT_EVERY = getattr(config, "STAGE10_1_CHECKPOINT_EVERY_STEPS", 100)
KEEP_EPOCH_CKPTS = getattr(config, "STAGE10_1_KEEP_EPOCH_CHECKPOINTS", True)
# Resume from a checkpoint without asking. False keeps the old behaviour --
# prompt on a TTY, auto-resume when stdin is not one. `--fresh` wins over both.
AUTO_RESUME = bool(getattr(config, "STAGE10_1_AUTO_RESUME", True))

STAGE = "stage10_1"
# The checkpoint format, the atomic write and the filenames now live in
# stage10_lineage, so that config.STAGE10_SHARED_OUTPUT can redirect all three
# Stage 10 scripts at ONE continuable checkpoint. With the flag off these are
# exactly the names and the payload this file always used.
CKPT_NAME = lineage._PRIVATE_NAMES[STAGE][0]
META_NAME = lineage._PRIVATE_NAMES[STAGE][1]
CKPT_FORMAT = lineage.CKPT_FORMAT


# ════════════════════════════════════════════════════════════════════════════
#  THE COMPOSITE LOSS, COMPOSED FROM A PRE-COMPUTED MEASUREMENT
# ════════════════════════════════════════════════════════════════════════════

def compose_stage10_loss(
    measured: Dict[str, Optional[float]],
) -> Tuple[float, Dict[str, float]]:
    """
    Stage 10's composite loss, but reading a measurement dict instead of
    running RDKit itself.

    This exists because the measurement has to happen in a worker process while
    the composition has to happen here: the pool returns raw
    compute_property_components dicts (with None preserved, so "not measurable"
    stays distinguishable from a real 0.0), and this turns one into the same
    (loss, breakdown) pair s10.compute_stage10_loss returns.

    It is the one piece of Stage 10 that is restated rather than imported --
    Stage 10 fuses measurement and composition into a single function and is
    being left untouched as the reference implementation, so there is no
    seam to import. Restating a formula invites drift, so the drift is made
    impossible to miss instead of merely unlikely: _run_self_test asserts this
    function and s10.compute_stage10_loss agree to zero tolerance, on valid
    molecules, invalid strings, self-comparisons and unmeasurable descriptors.
    If anyone edits the weights or the shape of the loss in Stage 10 without
    editing it here, that test fails.
    """
    if not measured["valid"]:
        return LOSS_INVALID, {
            "valid": 0.0, "qed": W_QED, "sa": W_SA,
            "novelty": W_NOVELTY, "tox_alert": W_TOX_ALERT,
            "invalid_penalty": W_VALID,
        }

    qed = measured["qed"]
    l_qed = W_QED * (1.0 - qed) if qed is not None else W_QED

    sa_raw = measured["sa_raw"]
    l_sa = (W_SA * min(max((sa_raw - 1.0) / 9.0, 0.0), 1.0)
            if sa_raw is not None else W_SA)

    nov = measured["novelty"]
    l_novelty = W_NOVELTY * (1.0 - nov) if nov is not None else W_NOVELTY

    alert = measured["any_alert"]
    l_alert = W_TOX_ALERT * alert if alert is not None else W_TOX_ALERT

    total = l_qed + l_sa + l_novelty + l_alert
    return float(total), {
        "valid": 1.0, "qed": l_qed, "sa": l_sa,
        "novelty": l_novelty, "tox_alert": l_alert, "invalid_penalty": 0.0,
    }


# ════════════════════════════════════════════════════════════════════════════
#  ONE TRAINING STEP, WITH SCORING LIFTED OUT TO THE POOL
# ════════════════════════════════════════════════════════════════════════════

def stage10_1_batch_loss(
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Stage 10's stage10_batch_loss with the RDKit work batched into one pool
    call. Returns the identical (loss, stats) pair, to the bit.

    Stage 10 interleaves the two costs per molecule:

        for molecule:  forward -> sample K -> [decode + score] x K -> CE

    which means the pool would be handed 16 items at a time, and the IPC would
    eat the parallelism. Here the same work is reordered into three phases so
    the pool sees the whole batch at once:

        phase 1  for molecule: forward -> sample K -> decode K     (GPU, serial)
        phase 2  measure all B x K candidates                      (pool)
        phase 3  for molecule: compose -> select best -> CE        (here)

    Reordering is safe because nothing in phase 2 or 3 touches a random number
    or a model weight: compute_property_components is a pure function of
    (generated, parent), pool.map preserves input order, and the sampling in
    phase 1 consumes the torch RNG in exactly the order Stage 10 consumes it
    (one _sample_candidates call per molecule, in batch order). The floating
    point ops that build the loss run in the same sequence on the same values,
    so the result is not merely equivalent but identical.

    Phase 1 holds every molecule's mask logits alive until phase 3. That is not
    a new memory cost: Stage 10 already accumulates `total = total + mol_loss`
    across the whole batch before calling backward, so the full autograd graph
    for all B molecules is resident either way.
    """
    variant             = (variant or VARIANT).lower()
    k_cand              = K_CANDIDATES        if k_cand is None else k_cand
    top_k               = TOP_K               if top_k is None else top_k
    temperature         = TEMPERATURE         if temperature is None else temperature
    fallback_weight     = FALLBACK_WEIGHT     if fallback_weight is None else fallback_weight
    unlikelihood_weight = UNLIKELIHOOD_WEIGHT if unlikelihood_weight is None else unlikelihood_weight

    stats: Dict[str, float] = {
        "n": 0.0, "fallback": 0.0, "cand_valid": 0.0, "cand_total": 0.0,
        "best_loss": 0.0, "unlikelihood": 0.0, "skipped": 0.0, "best_valid": 0.0,
    }
    for key in LOSS_TERMS:
        stats[key] = 0.0

    # ── phase 1: forward, sample, decode ──────────────────────────────────
    pending: List[dict] = []
    flat: List[Tuple[str, str]] = []
    for masked_smi, parent_smi in batch:
        enc = tokenizer(masked_smi, return_tensors="pt", truncation=True,
                        max_length=MAX_MODEL_TOKENS).to(device)
        ids = enc["input_ids"][0]
        mask_pos = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if mask_pos.numel() == 0:
            stats["skipped"] += 1
            continue

        logits = model(**enc).logits[0]                     # [L, V], carries grad
        at_masks = logits[mask_pos]                         # [n_masks, V]

        cands = s10._sample_candidates(at_masks, k_cand, top_k, temperature)

        smiles: List[str] = []
        for row in cands:
            filled = ids.clone()
            filled[mask_pos] = row
            smiles.append(
                tokenizer.decode(filled, skip_special_tokens=True).replace(" ", ""))

        pending.append({
            "masked": masked_smi, "parent": parent_smi,
            "at_masks": at_masks, "mask_pos": mask_pos, "cands": cands,
        })
        flat.extend((s, parent_smi) for s in smiles)

    if not pending:
        return torch.zeros((), device=device, requires_grad=True), stats

    # ── phase 2: every candidate in the batch, measured at once ───────────
    measured = pool.measure(flat)

    # ── phase 3: compose, select, differentiate ───────────────────────────
    total = torch.zeros((), device=device)
    n_weighted = 0.0
    cursor = 0

    for item in pending:
        cands    = item["cands"]
        at_masks = item["at_masks"]
        mask_pos = item["mask_pos"]

        best_loss, best_ids, best_comps = None, None, None
        invalid_rows: List[torch.Tensor] = []
        for row in cands:
            loss_c, comps = compose_stage10_loss(measured[cursor])
            cursor += 1
            stats["cand_total"] += 1
            if comps["valid"]:
                stats["cand_valid"] += 1
            else:
                invalid_rows.append(row)
            if best_loss is None or loss_c < best_loss:
                best_loss, best_ids, best_comps = loss_c, row, comps

        used_fallback = (best_comps is None) or (not best_comps["valid"])
        if used_fallback:
            tgt = s10.parent_target_ids(item["masked"], item["parent"], tokenizer)
            if tgt is None or len(tgt) != mask_pos.numel():
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
            bad = torch.stack(invalid_rows)
            keep = bad != best_ids.unsqueeze(0)
            if keep.any():
                probs = log_probs.exp()
                p_bad = probs.gather(1, bad.t()).t()
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
#  DETERMINISTIC EPOCH ORDER
# ════════════════════════════════════════════════════════════════════════════

def _epoch_order(n_pairs: int, epoch: int, seed: int = SEED) -> List[int]:
    """
    The permutation Stage 10 would be training on at `epoch`, rebuilt from
    scratch. Returns indices into the ORIGINAL pairs list.

    Two problems are solved at once here.

    RESUMABILITY. Stage 10 shuffles the pairs list IN PLACE, once per epoch,
    from an rng seeded random.Random(42 + start_epoch). That is not a
    reproducible address: the permutation at epoch 3 depends on every shuffle
    since the run began, so there is no way to ask "which molecules are in
    batch 812 of epoch 3" without having run epochs 1 and 2 first. Resuming
    mid-epoch requires exactly that question to be answerable, so the order is
    made a pure function of (n_pairs, epoch, seed) instead.

    COMPARABILITY. It reproduces Stage 10's permutation rather than inventing a
    new one. random.shuffle is a Fisher-Yates over POSITIONS and consumes the
    same draws whatever the list holds, so composing `epoch` shuffles of an
    index array from Random(seed + 1) yields precisely the arrangement Stage
    10's cumulative in-place shuffles produce at that epoch, for a fresh run
    (start_epoch = 1, hence seed + 1). Stage 10.1 therefore trains on the same
    molecules in the same batches in the same order, which is what makes a
    10 vs 10.1 timing comparison a comparison of execution and nothing else.

    Cost is O(epoch * n_pairs) with epoch <= 4 and n_pairs ~65k -- a few
    milliseconds, paid once per epoch, in exchange for random access.
    """
    rng = random.Random(seed + 1)
    order = list(range(n_pairs))
    for _ in range(max(epoch, 1)):
        rng.shuffle(order)
    return order


def _batches_for_epoch(n_pairs: int, epoch: int, batch_size: int,
                       seed: int = SEED) -> List[List[int]]:
    """The epoch's batches as index lists. Pure function of its arguments."""
    order = _epoch_order(n_pairs, epoch, seed)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


# ════════════════════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ════════════════════════════════════════════════════════════════════════════

def _ckpt_path(save_dir: str) -> str:
    """The rolling checkpoint. In shared mode this is the family's lineage
    file, which is what lets a run started under Stage 10 or Stage 10.2 be
    continued here."""
    return lineage.ckpt_path(save_dir, STAGE)


def _meta_path(save_dir: str) -> str:
    return lineage.meta_path(save_dir, STAGE)


def _fingerprint(variant: str, batch_size: int, k_cand: int,
                 num_epochs: int, n_pairs: int, seed: int) -> Dict[str, object]:
    """
    The settings a mid-epoch resume depends on.

    batch_index is an offset into a batch list built by _batches_for_epoch, so
    it only means anything if n_pairs, batch_size and seed are what they were
    when it was written. k_cand, variant and num_epochs do not affect the
    batching but do change what is being trained, so a mismatch there is worth
    reporting even though it is recoverable.
    """
    return {"variant": variant, "batch_size": batch_size, "k_cand": k_cand,
            "num_epochs": num_epochs, "n_pairs": n_pairs, "seed": seed}


# The random streams and their restoration now live in stage10_lineage, so all
# three Stage 10 scripts capture the SAME set -- a checkpoint written by one is
# only continuable by another if they agree on what "the RNG state" is. Aliased
# rather than re-exported under new names so every existing call site, and the
# self-test, keep working unchanged.
def _rng_state() -> Dict[str, object]:
    """Every random stream the training loop consumes; see
    stage10_lineage.rng_state for why torch's is the one that matters."""
    return lineage.rng_state()


def _restore_rng(state: Optional[Dict[str, object]]) -> None:
    """Put the streams back. Missing or unusable state is a warning, not a
    failure: continuing with the right weights and a fresh RNG is far better
    than refusing to resume a ten-hour run."""
    lineage.restore_rng(state, log=tqdm.write)


def _execution_profile(workers: int, device: str = "cpu") -> Dict[str, object]:
    """
    How THIS stage executes, for the provenance log.

    Fixed by construction rather than configurable: Stage 10.1's whole claim is
    that it keeps the per-molecule forward and the per-molecule sampler in
    fp32, which is what makes it bit-identical to Stage 10. amp=None and
    batched=False are that claim, written down where a later reader -- or a
    Stage 10.2 resume -- can check it.
    """
    return lineage.execution_profile(
        STAGE, amp=None, batched=False, workers=int(workers), device=device)


def save_checkpoint(
    save_dir:    str,
    model,
    optimizer:   torch.optim.Optimizer,
    epoch:       int,
    batch_index: int,
    global_step: int,
    history:     Dict[str, list],
    agg:         Dict[str, float],
    n_steps:     int,
    fingerprint: Dict[str, object],
    provenance:  Optional[List[dict]] = None,
    profile:     Optional[Dict[str, object]] = None,
) -> List[dict]:
    """
    Write the complete resumable state ATOMICALLY, as one rolling file, and
    return the updated provenance list.

    `batch_index` is the index of the NEXT batch to run inside `epoch`, so a
    resume needs no "did the last one finish" reasoning: it starts there.

    `agg` and `n_steps` are the running per-epoch aggregates. Without them a
    resume mid-epoch would compute that epoch's history row from only the
    batches that ran AFTER the restart, and the curve would show a step change
    at every resume that has nothing to do with the model.

    The write itself -- temp file plus os.replace, atomic on POSIX and on
    Windows -- and the JSON sidecar now live in stage10_lineage.save_state, so
    that all three Stage 10 scripts write ONE format and a run begun under any
    of them can be finished under any other. See stage10_lineage for why the
    atomicity matters at 93 MB per write, and for what the provenance list is
    for.
    """
    return lineage.save_state(
        save_dir, STAGE,
        trainable   = s10._trainable_state(model),
        optimizer   = optimizer.state_dict(),
        epoch       = epoch,
        batch_index = batch_index,
        global_step = global_step,
        history     = history,
        agg         = agg,
        n_steps     = n_steps,
        fingerprint = fingerprint,
        rng         = _rng_state(),
        profile     = profile or _execution_profile(0),
        provenance  = provenance,
    )


def load_checkpoint(save_dir: str) -> Optional[dict]:
    """The rolling checkpoint, or None. A corrupt file reports and returns
    None rather than raising, so a damaged checkpoint costs the run its
    progress and not its ability to start."""
    return lineage.load_state(save_dir, STAGE, log=tqdm.write)


def _ask_resume(save_dir: str, ckpt: dict) -> bool:
    """
    Ask whether to resume, and auto-resume whenever stdin is not a TTY.

    Same convention and the same reasoning as stage1_9._ask_resume: a Colab
    `!python` cell, a SLURM batch job or a nohup'd run has no terminal, so the
    prompt could only raise EOFError or block forever, and an unattended job
    that was restarted overwhelmingly wants to continue. Continuing is also the
    non-destructive answer -- "no" starts over and the first save overwrites
    the checkpoint.

    config.STAGE10_1_AUTO_RESUME = True skips the question on a TTY as well,
    for the same reason: on a run you restart by hand the answer is always yes.
    `--fresh` still starts over, and an explicit auto_resume= argument from the
    caller never reaches here at all.
    """
    fp = ckpt.get("fingerprint", {})
    print(f"\n  Checkpoint found in : {save_dir}")
    print(f"  Epoch               : {ckpt['epoch']}")
    print(f"  Next batch in epoch : {ckpt['batch_index']}")
    print(f"  Global step         : {ckpt['global_step']}")
    print(f"  Variant             : {fp.get('variant', '?')}")
    if ckpt.get("saved_at"):
        print(f"  Saved               : "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ckpt['saved_at']))}")

    if AUTO_RESUME:
        print("  config.STAGE10_1_AUTO_RESUME is on -- resuming automatically. "
              "Pass --fresh (or delete the checkpoint) to start over instead.")
        return True
    if not sys.stdin.isatty():
        print("  Non-interactive session -- resuming automatically. "
              "Pass --fresh (or delete the checkpoint) to start over instead.")
        return True
    while True:
        try:
            ans = input("  Resume from checkpoint? [Y/n]: ").strip().lower()
        except EOFError:
            print("  No input available -- resuming automatically.")
            return True
        if ans in ("", "y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please enter Y or N.")


class _InterruptGuard:
    """
    Turn SIGINT into a flag the training loop can act on.

    Saving from inside the handler is the obvious implementation and the wrong
    one: the signal can arrive in the middle of the 93 MB torch.save this very
    mechanism triggers, and re-entering it truncates the file. So the handler
    only records that a stop was requested; the loop finishes the batch it is
    on, writes one clean checkpoint and returns.

    The second Ctrl-C restores the default handler and re-raises, so a genuinely
    stuck run is still killable with a second press rather than trapping the
    user in a process that will not die.
    """

    def __init__(self) -> None:
        self.requested = False
        self._previous = None

    def __enter__(self) -> "_InterruptGuard":
        try:
            self._previous = signal.signal(signal.SIGINT, self._handle)
        except (ValueError, OSError):     # not the main thread; nothing to do
            self._previous = None
        return self

    def _handle(self, signum, frame):
        if self.requested:
            # Second press: give the terminal back its normal behaviour.
            signal.signal(signal.SIGINT, self._previous or signal.SIG_DFL)
            raise KeyboardInterrupt
        self.requested = True
        # tqdm.write, never print: this fires mid-loop with a bar on
        # screen, and a bare print() emits a newline the bar does not
        # know about, leaving an orphaned copy of it behind on every
        # terminal that draws one. tqdm.write clears the bar, writes the
        # line and redraws it. Stage 10.2 imports this guard rather than
        # restating it, so the same fix covers both scripts.
        tqdm.write("\n  Interrupt received -- finishing this batch, then saving "
                   "a checkpoint and stopping. Press Ctrl-C again to abort "
                   "now (losing progress since the last save).")

    def __exit__(self, *exc) -> None:
        if self._previous is not None:
            try:
                signal.signal(signal.SIGINT, self._previous)
            except (ValueError, OSError):
                pass


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_stage10_1_training(
    pairs:       List[Tuple[str, str]],
    save_dir:    str   = None,
    variant:     str   = None,
    num_epochs:  int   = NUM_EPOCHS,
    batch_size:  int   = BATCH_SIZE,
    lr:          float = LEARNING_RATE,
    grad_clip:   float = GRAD_CLIP,
    k_cand:      int   = K_CANDIDATES,
    workers:     int   = None,
    ckpt_every:  int   = None,
    seed:        int   = SEED,
    fresh:       bool  = False,
    auto_resume: Optional[bool] = None,
) -> Dict[str, list]:
    """
    Stage 10's supervised best-of-K loop, resumable at step granularity and
    scored across a process pool.

    Returns the same history dict Stage 10 returns, so _plot_history and every
    downstream reader work unchanged.

    `auto_resume` overrides the interactive prompt: True resumes silently,
    False ignores the checkpoint. Left as None it asks, falling back to
    resuming when stdin is not a terminal. A caller that already knows the
    answer needs this, because the prompt would otherwise block a driver
    script on a real TTY.
    """
    variant = (variant or VARIANT).lower()
    if variant not in ("a", "b"):
        raise ValueError(f"variant must be 'a' or 'b', got {variant!r}")
    save_dir = save_dir or lineage.resolve_save_dir(STAGE, variant)
    os.makedirs(save_dir, exist_ok=True)

    workers = (resolve_workers(getattr(config, "STAGE10_1_SCORING_WORKERS", "auto"))
               if workers is None else resolve_workers(workers))
    if ckpt_every is None:
        ckpt_every = CKPT_EVERY
    batch_size = int(batch_size)

    hw = get_profile()
    hw.apply()                        # torch thread count, TF32 where available

    # ── hold out the validation fold BEFORE the fingerprint is computed ───
    # Whole Bemis-Murcko groups, from the manifest every Stage 10 variant
    # shares, so len(pairs) below -- and therefore the resume fingerprint --
    # refers to the TRAINING fold only. A checkpoint written before the split
    # existed carries a different n_pairs and will correctly report a
    # batching change rather than silently resuming onto different data.
    pairs, val_pairs, _ = split.split_pairs(pairs)
    for line in split.describe(
            split.get_split([q for _, q in pairs + val_pairs]),
            (pairs, val_pairs, [])):
        tqdm.write(line)
    best_val = float("inf")

    history: Dict[str, list] = s10.new_history()

    def _fresh_agg() -> Dict[str, float]:
        a = {k: 0.0 for k in ("loss", "best_loss", "fallback", "cand_valid",
                              "cand_total", "n", "unlikelihood", "best_valid")}
        for key in LOSS_TERMS:
            a[key] = 0.0
        return a

    fp = _fingerprint(variant, batch_size, k_cand, num_epochs, len(pairs), seed)

    # ── resume ────────────────────────────────────────────────────────────
    ckpt = None if fresh else load_checkpoint(save_dir)
    start_epoch, start_batch, global_step = 1, 0, 0
    agg, n_steps = _fresh_agg(), 0
    provenance: List[dict] = []
    profile = _execution_profile(workers)
    resume_state = None

    if ckpt and (auto_resume if auto_resume is not None
                 else _ask_resume(save_dir, ckpt)):
        # In shared mode this checkpoint may have been written by Stage 10 or
        # Stage 10.2, under a different precision or a different sampler. The
        # weights continue either way; the banner is what stops that going
        # unrecorded. Nothing to print when the execution is unchanged.
        banner = lineage.lineage_banner(ckpt, profile)
        if banner:
            tqdm.write(banner)
        provenance = ckpt.get("provenance") or []
        old_fp = ckpt.get("fingerprint", {})
        # batch_index addresses a list built from these three numbers. If any
        # changed, the index points at different molecules, so the honest move
        # is to keep the weights and the optimizer and restart the epoch --
        # not to silently train on the wrong slice.
        batching_same = all(old_fp.get(k) == fp[k]
                            for k in ("n_pairs", "batch_size", "seed"))
        start_epoch  = int(ckpt["epoch"])
        global_step  = int(ckpt["global_step"])
        history      = s10.ensure_history_keys(ckpt["history"])
        resume_state = ckpt
        if batching_same:
            start_batch = int(ckpt["batch_index"])
            agg     = ckpt.get("agg") or _fresh_agg()
            n_steps = int(ckpt.get("n_steps", 0))
        else:
            changed = [k for k in ("n_pairs", "batch_size", "seed")
                       if old_fp.get(k) != fp[k]]
            tqdm.write(
                f"  Batching settings changed since the checkpoint ({', '.join(changed)}); "
                f"the saved batch index no longer addresses the same molecules. "
                f"Keeping the weights and optimizer, restarting epoch {start_epoch}.")
            start_batch, agg, n_steps = 0, _fresh_agg(), 0
        for k in ("variant", "k_cand"):
            if old_fp.get(k) != fp[k]:
                tqdm.write(f"  NOTE: {k} changed "
                           f"({old_fp.get(k)!r} -> {fp[k]!r}) since the checkpoint.")
        tqdm.write(f"\n  Resuming at epoch {start_epoch}, batch {start_batch}, "
                   f"step {global_step}.")
    else:
        tqdm.write("\n  Starting a fresh training run.")

    n_batches_per_epoch = (len(pairs) + batch_size - 1) // batch_size
    if start_epoch > num_epochs or (start_epoch == num_epochs
                                    and start_batch >= n_batches_per_epoch):
        tqdm.write("  Training already complete (all epochs done).")
        return history

    # ── model + optimizer ─────────────────────────────────────────────────
    tokenizer, model, device = s10.load_model_last_layers()
    if resume_state is not None:
        # strict=False by necessity: the checkpoint holds ONLY the unfrozen
        # tensors (s10._trainable_state), so every frozen weight is reported
        # missing and is supposed to be -- it came from the pretrained model.
        model.load_state_dict(resume_state["trainable"], strict=False)
        tqdm.write(f"  Restored {len(resume_state['trainable'])} trainable "
                   f"tensor(s) from the checkpoint.")
    model.train()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    if resume_state is not None and resume_state.get("optimizer"):
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
            tqdm.write("  Optimizer state restored (Adam moments preserved).")
        except Exception as e:
            tqdm.write(f"  Could not restore optimizer state ({e}); "
                       f"continuing with a fresh Adam.")
    _restore_rng(resume_state.get("rng") if resume_state else None)
    if resume_state is None:
        # Seed the candidate-sampling stream so a fresh run is reproducible and
        # a resumed one has something coherent to continue.
        torch.manual_seed(seed)
    del resume_state

    tqdm.write(
        f"\n  Variant {variant} : "
        + ("CE toward best candidate + unlikelihood on invalid ones"
           if variant == "a" else "CE toward best candidate only")
        + f"\n  K candidates    : {k_cand}"
        + f"\n  batch size      : {batch_size}  "
          f"({n_batches_per_epoch} batches/epoch, same as Stage 10)"
        + f"\n  scoring workers : "
        + (f"{workers}  ({k_cand * batch_size} candidates/batch across the pool)"
           if workers else "serial (no pool)")
        + f"   [config.STAGE10_1_SCORING_WORKERS]"
        + f"\n  checkpoint      : "
        + (f"every {ckpt_every} steps + every epoch"
           if ckpt_every else "every epoch only (mid-epoch saves disabled)")
        + f"   [config.STAGE10_1_CHECKPOINT_EVERY_STEPS]"
        + f"\n  hardware        : {hw.environment}, {hw.cpu_count} CPU "
          f"({hw.cpu_source}), device {device}"
        + f"\n  loss (valid)    : {W_QED}*(1-QED) + {W_SA}*(SA-1)/9 "
          f"+ {W_NOVELTY}*similarity + {W_TOX_ALERT}*alert   -> at most {S_WORST_VALID}"
        + f"\n  loss (invalid)  : {LOSS_INVALID}  (= {S_WORST_VALID} + w_valid {W_VALID})"
        + f"\n  fallback weight : {FALLBACK_WEIGHT}  (parent-reconstruction steps)"
    )

    total_remaining = ((num_epochs - start_epoch + 1) * n_batches_per_epoch
                       - start_batch)
    # A live bar on a TTY, a status line every ckpt_every batches when there is
    # no TTY to draw one on (Colab's `!python`, nohup, a piped log). Tying the
    # cadence to the checkpoint interval means each line marks roughly one
    # "this much is now safely on disk". See stage10_lineage.Progress.
    pbar = lineage.Progress(total=max(total_remaining, 0),
                            desc=f"Stage 10.1{variant} training",
                            every=ckpt_every or 100)

    stopped_early = False
    epoch = start_epoch

    with ScoringPool(workers) as pool, _InterruptGuard() as guard:
        for epoch in range(start_epoch, num_epochs + 1):
            batches = _batches_for_epoch(len(pairs), epoch, batch_size, seed)
            first = start_batch if epoch == start_epoch else 0
            if first == 0:
                agg, n_steps = _fresh_agg(), 0

            for bi in range(first, len(batches)):
                batch = [pairs[i] for i in batches[bi]]
                optimizer.zero_grad()
                loss, stats = stage10_1_batch_loss(
                    batch, tokenizer, model, device, pool,
                    variant=variant, k_cand=k_cand)

                if stats["n"] == 0:
                    pbar.update(1)
                    continue

                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip)
                optimizer.step()

                agg["loss"] += float(loss.detach())
                for k in ("best_loss", "fallback", "cand_valid", "cand_total",
                          "n", "unlikelihood", "best_valid"):
                    agg[k] += stats[k]
                for key in LOSS_TERMS:
                    agg[key] += stats[key]
                n_steps += 1
                global_step += 1

                pbar.set_postfix_str(
                    f"ep={epoch}/{num_epochs}  loss={float(loss.detach()):.4f}  "
                    f"best={stats['best_loss'] / max(stats['n'], 1):.3f}  "
                    f"fallback={stats['fallback'] / max(stats['n'], 1):.0%}  "
                    f"cand_valid={stats['cand_valid'] / max(stats['cand_total'], 1):.0%}",
                    refresh=True)
                pbar.update(1)

                # ONE read of the flag per batch, deliberately. Reading it
                # twice leaves a window: the signal lands after the save test
                # has already seen False and before the break test sees True,
                # so the loop exits without writing the checkpoint that the
                # interrupt was supposed to produce.
                stop_now = guard.requested
                if stop_now or (ckpt_every and global_step % ckpt_every == 0):
                    provenance = save_checkpoint(
                        save_dir, model, optimizer, epoch, bi + 1,
                        global_step, history, agg, n_steps, fp,
                        provenance=provenance, profile=profile)
                if stop_now:
                    stopped_early = True
                    break

            if stopped_early:
                break

            # ── epoch boundary ────────────────────────────────────────────
            n_mol = max(agg["n"], 1)
            row = {
                "epoch": epoch,
                "loss_mean": agg["loss"] / max(n_steps, 1),
                "best_loss_mean": agg["best_loss"] / n_mol,
                "fallback_rate": agg["fallback"] / n_mol,
                "cand_valid_rate": agg["cand_valid"] / max(agg["cand_total"], 1),
                "best_valid_rate": agg["best_valid"] / n_mol,
                "unlikelihood_mean": agg["unlikelihood"] / n_mol,
                **{key: agg[key] / n_mol for key in LOSS_TERMS},
            }
            row["tox_alert_rate"] = s10.alert_rate_from_loss(row["tox_alert"])
            # `pool` is in scope here, which is why the closure is built at
            # the call site rather than beside the fold: this loss needs it.
            row.update(split.validation_pass(
                val_pairs,
                lambda b: stage10_1_batch_loss(b, tokenizer, model, device,
                                               pool, variant=variant,
                                               k_cand=k_cand),
                model, LOSS_TERMS, batch_size=batch_size, seed=seed))
            for k in s10.HISTORY_KEYS:
                if k in row:
                    history[k].append(row[k])
                elif len(history[k]) < len(history["epoch"]):
                    history[k].append(float("nan"))

            tqdm.write(
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
                + ("\n" + split.format_val_line(row)
                   if row.get("val_n") else "")
            )

            # MODEL SELECTION on the held-out fold, saved the moment it
            # improves. A run stopped early then leaves the BEST epoch on
            # disk rather than the last one, and run_stage10_eval -- which
            # loads from save_dir -- scores the selected epoch.
            if split.SELECT_BEST_VAL and split.is_best_val(history):
                best_val = history["val_best_loss_mean"][-1]
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                tqdm.write(f"    new best held-out loss {best_val:.4f} "
                           f"-- model saved to {save_dir}")

            if KEEP_EPOCH_CKPTS:
                ep_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
                # Stage 10's per-epoch format exactly, so its checkpoints and
                # these are interchangeable for inspection and for
                # load_model_last_layers(checkpoint=...).
                torch.save({"trainable": s10._trainable_state(model)}, ep_path)
                tqdm.write(f"  Epoch snapshot saved -> {ep_path}")

            # batch_index = 0 of the NEXT epoch: this epoch is finished.
            provenance = save_checkpoint(
                save_dir, model, optimizer, epoch + 1, 0,
                global_step, history, _fresh_agg(), 0, fp,
                provenance=provenance, profile=profile)
            agg, n_steps = _fresh_agg(), 0

    pbar.close()

    if stopped_early:
        tqdm.write(
            f"\n  Stopped at epoch {epoch}, step {global_step}. State saved to "
            f"{_ckpt_path(save_dir)}.\n"
            f"  Re-run the same command to continue from exactly here.")
        return history

    # Skipped when an epoch was selected on the held-out fold: those
    # weights are already in save_dir, and re-saving here would overwrite
    # the SELECTED model with the LAST one, silently undoing the selection.
    if not (split.SELECT_BEST_VAL and best_val < float("inf")):
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        tqdm.write(f"\n  Final model saved to : {save_dir}")
    else:
        tqdm.write(f"\n  Final model is the epoch with the lowest held-out "
                   f"loss ({best_val:.4f}), already in : {save_dir}")
    # ".1a" not "1a": these two interpolate the variant directly into the
    # filename and the title, so "1a" would read "stage101a".
    s10._plot_history(history, save_dir, f".1{variant}")
    s10._plot_tox_alert_rate(history, save_dir, f".1{variant}")
    s10._plot_validation_properties(history, save_dir, f".1{variant}")
    return history


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(variant: str = None, max_pairs_per_source: int = None,
         sample_seed: int = None, workers: int = None,
         fresh: bool = False, num_epochs: int = None) -> None:
    variant = (variant or VARIANT).lower()
    save_dir = lineage.resolve_save_dir(STAGE, variant)
    print("\n" + "=" * 62)
    print(f"STAGE 10.1{variant.upper()} -- SUPERVISED BEST-OF-K, RESUMABLE + POOLED")
    print("=" * 62)
    print(f"""
  Stage 10's objective, unchanged. What differs is execution:
    - resumable at every {CKPT_EVERY or 'epoch'} optimizer step(s), with the
      Adam moments and the RNG streams, so a Ctrl-C or a dropped runtime
      costs minutes instead of an epoch
    - the {K_CANDIDATES} x batch RDKit candidate measurements per step run
      across a process pool instead of serially

  loss(valid)   = {W_QED}*(1-QED) + {W_SA}*(SA-1)/9 + {W_NOVELTY}*similarity + {W_TOX_ALERT}*alert
  loss(invalid) = {LOSS_INVALID}   (worst valid = {S_WORST_VALID}, plus w_valid = {W_VALID})
  variant       = {variant}  ({"CE + unlikelihood on invalid" if variant == "a" else "CE only"})
  mask percent  = {config.MASK_PERCENT}%   (shared with Stage 9 and Stage 10)""")
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

    history = run_stage10_1_training(
        pairs=pairs, save_dir=save_dir, variant=variant,
        workers=workers, fresh=fresh,
        **({} if num_epochs is None else {"num_epochs": num_epochs}))

    print("\n" + "=" * 62)
    print(f"  Stage 10.1{variant} training complete.")
    if history.get("epoch"):
        print(f"  Final training loss        : {history['loss_mean'][-1]:.4f}")
        print(f"  Final candidate validity   : {history['cand_valid_rate'][-1]:.1%}")
        print(f"  Final parent-fallback rate : {history['fallback_rate'][-1]:.1%}")
        print(f"  Final toxicity alert rate  : {history['tox_alert_rate'][-1]:.1%}")
    print(f"  Model : {save_dir}")
    print("=" * 62)

    if os.path.isfile(os.path.join(save_dir, "config.json")):
        s10.run_stage10_eval(save_dir, f".1{variant}",
                             max_pairs_per_source, sample_seed)
    else:
        print("  Run stopped before the final model was written -- skipping the "
              "evaluation pass. Re-run to finish training first.")


def _parse_args(argv: list) -> tuple:
    """--variant a|b, --limit N ("none"/"all"/0 = uncapped), --seed N,
    --workers N, --fresh, --shared / --separate."""
    if "--shared" in argv:
        lineage.set_shared_override(True)
    elif "--separate" in argv:
        lineage.set_shared_override(False)
    variant = None
    epochs = None
    limit = seed = workers = None
    for flag in ("--variant", "--limit", "--seed", "--workers",
                 "--epochs"):
        if flag not in argv:
            continue
        idx = argv.index(flag)
        if idx + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        raw = argv[idx + 1]
        if flag == "--epochs":
            epochs = int(raw)
        elif flag == "--variant":
            variant = raw.lower()
        elif flag == "--limit":
            limit = 0 if raw.lower() in ("none", "all", "0") else int(raw)
        elif flag == "--workers":
            workers = int(raw)
        else:
            seed = int(raw)
    return variant, limit, seed, workers, ("--fresh" in argv), epochs


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    """
    Pins the three claims this file makes, in order of what would hurt most if
    wrong.

      1. compose_stage10_loss agrees with Stage 10's compute_stage10_loss to
         ZERO tolerance. This is the restated-formula risk; if it drifts,
         Stage 10.1 silently optimises a different objective.
      2. The pooled batch loss is BIT-IDENTICAL to Stage 10's serial one at the
         same seed, serial and pooled alike. This is what makes 10 vs 10.1 a
         pure execution comparison.
      3. _epoch_order reproduces Stage 10's permutation, and a mid-epoch
         interrupt round-trips: weights, Adam moments, aggregates and step
         count all come back, and resuming a finished run is a no-op.
    """
    import tempfile
    import warnings
    warnings.filterwarnings("ignore")

    from stage10_self_test import _build_pairs

    print("\n" + "=" * 62)
    print("  STAGE 10.1 SELF-TEST")
    print("=" * 62)

    # ── 1. the restated loss matches Stage 10's, exactly ──────────────────
    from stage9_1_scoring_worker import score_one
    parent = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    probes = [
        ("not_a_smiles(((", parent),          # invalid
        (parent, parent),                     # self: full novelty penalty
        ("CC(=O)Nc1ccc(O)cc1", parent),       # a genuinely different molecule
        ("CC(=O)Oc1ccccc1C(=O)O", parent),
        ("", parent),                         # empty: invalid
        ("[Xe]", parent),                     # parses, descriptors are odd
    ]
    for gen, par in probes:
        ref_loss, ref_comps = s10.compute_stage10_loss(gen, par)
        got_loss, got_comps = compose_stage10_loss(score_one((gen, par)))
        assert got_loss == ref_loss, f"{gen!r}: {got_loss} != {ref_loss}"
        assert got_comps == ref_comps, f"{gen!r}: {got_comps} != {ref_comps}"
    print(f"  [1] compose_stage10_loss == s10.compute_stage10_loss on "
          f"{len(probes)} probes, exactly   OK")

    # ── 2. pooled batch loss is identical to Stage 10's serial one ────────
    tokenizer, model, device = s10.load_model_last_layers()
    model.train()
    PAIRS = _build_pairs(tokenizer)

    torch.manual_seed(1234)
    ref_loss, ref_stats = s10.stage10_batch_loss(
        PAIRS, tokenizer, model, device, variant="a", k_cand=6)

    for n_workers, label in ((0, "serial"), (2, "2-worker pool")):
        torch.manual_seed(1234)
        with ScoringPool(n_workers) as pool:
            got_loss, got_stats = stage10_1_batch_loss(
                PAIRS, tokenizer, model, device, pool, variant="a", k_cand=6)
        assert float(got_loss) == float(ref_loss), (
            f"{label}: batch loss {float(got_loss)!r} != Stage 10's "
            f"{float(ref_loss)!r}")
        assert got_stats == ref_stats, f"{label}: stats differ\n{got_stats}\n{ref_stats}"
        assert got_loss.requires_grad, f"{label}: loss must carry a gradient"
    print(f"  [2] pooled loss == Stage 10 serial loss to zero tolerance "
          f"({float(ref_loss):.6f}), serial and pooled   OK")

    # Variant b must still differ from a, or the comparison measures nothing.
    torch.manual_seed(1234)
    with ScoringPool(0) as pool:
        lb, _ = stage10_1_batch_loss(PAIRS, tokenizer, model, device, pool,
                                     variant="b", k_cand=6)
    assert float(lb) < float(ref_loss), "10a must exceed 10b by the unlikelihood term"
    print(f"  [2b] variant a {float(ref_loss):.4f} > variant b {float(lb):.4f}   OK")

    # ── 3. epoch order matches Stage 10's cumulative shuffle ──────────────
    n = 97
    for n_ep in (1, 2, 3, 4):
        items = list(range(n))
        rng = random.Random(SEED + 1)                # Stage 10: 42 + start_epoch
        for _ in range(n_ep):
            rng.shuffle(items)                       # in place, cumulative
        assert _epoch_order(n, n_ep, SEED) == items, (
            f"epoch {n_ep}: _epoch_order does not reproduce Stage 10's shuffle")
    assert sorted(_epoch_order(n, 3, SEED)) == list(range(n)), "not a permutation"
    print("  [3] _epoch_order reproduces Stage 10's permutation for epochs 1-4   OK")

    # ── 4. interrupt mid-epoch, resume, and land in the same place ────────
    with tempfile.TemporaryDirectory() as td:
        pairs = list(PAIRS) * 3                      # 12 pairs -> 6 batches of 2
        kw = dict(pairs=pairs, save_dir=td, variant="b", num_epochs=1,
                  batch_size=2, k_cand=4, workers=0, ckpt_every=1,
                  auto_resume=True)

        # Stand in for a Ctrl-C at a known point. The loop reads
        # guard.requested exactly once per batch, so a read counter stops the
        # run deterministically after the 3rd batch of 6 -- genuinely
        # mid-epoch, which is the case Stage 10 cannot represent at all.
        STOP_AFTER = 3
        real_guard = _InterruptGuard

        class _StopAfterNBatches(_InterruptGuard):
            reads = 0

            @property
            def requested(self):                     # type: ignore[override]
                type(self).reads += 1
                # >=, not >: the flag is read once per batch AFTER that batch's
                # step, so read number N fires on batch index N-1 and the
                # checkpoint records batch_index = N. STOP_AFTER batches run.
                return type(self).reads >= STOP_AFTER

            @requested.setter
            def requested(self, value):
                pass                                 # __init__ assigns False

        globals()["_InterruptGuard"] = _StopAfterNBatches
        try:
            run_stage10_1_training(**kw)
        finally:
            globals()["_InterruptGuard"] = real_guard

        mid = load_checkpoint(td)
        assert mid is not None, "an interrupted run must leave a checkpoint"
        assert mid["epoch"] == 1, f"expected epoch 1, got {mid['epoch']}"
        assert mid["batch_index"] == STOP_AFTER, (
            f"expected a stop at batch {STOP_AFTER}, got {mid['batch_index']}")
        assert 0 < mid["batch_index"] < 6, "the stop must be genuinely mid-epoch"
        assert mid["optimizer"]["state"], "Adam moments must be in the checkpoint"
        assert mid["n_steps"] == mid["global_step"] == STOP_AFTER
        assert mid["rng"]["torch"] is not None, "RNG state must be checkpointed"
        assert mid["history"]["epoch"] == [], "a partial epoch writes no history row"
        assert not os.path.isfile(os.path.join(td, "config.json")), (
            "an interrupted run must NOT write the final model")
        # Adam moments must be non-trivial, or "optimizer restored" is a lie.
        moments = [v for v in mid["optimizer"]["state"].values() if "exp_avg" in v]
        assert moments and any(float(v["exp_avg"].abs().sum()) > 0 for v in moments),             "Adam exp_avg must be populated by the time we checkpoint"
        print(f"  [4] interrupt left a checkpoint at epoch 1 batch "
              f"{mid['batch_index']}/6, with populated Adam moments and RNG   OK")

        # Resume: must pick up at that batch, finish the epoch, write the model.
        hist = run_stage10_1_training(**kw)
        assert hist["epoch"] == [1], f"resume should finish epoch 1, got {hist['epoch']}"
        assert os.path.isfile(os.path.join(td, "config.json")),             "a finished run must save the model in HF format for the eval pass"
        assert os.path.isfile(os.path.join(td, "epoch_001.pt")),             "the per-epoch snapshot must still be written"
        done = load_checkpoint(td)
        assert done["epoch"] == 2 and done["batch_index"] == 0, (
            f"a finished epoch must point at the next one, got "
            f"epoch {done['epoch']} batch {done['batch_index']}")
        assert done["global_step"] == 6, (
            f"6 batches must have run in total across the interrupt, got "
            f"{done['global_step']}")
        print(f"  [5] resumed from batch {STOP_AFTER}, ran the remaining "
              f"{6 - STOP_AFTER}, wrote the final model   OK")

        # Resuming a completed run must be a no-op, not a retrain.
        again = run_stage10_1_training(**kw)
        assert again["epoch"] == [1], "resume must not re-run a finished epoch"
        assert load_checkpoint(td)["global_step"] == 6, "a no-op must not train"
        print("  [6] resuming a finished run is a no-op   OK")

    # ── 7. a resumed run == an uninterrupted one, weight for weight ───────
    # The whole point of the feature, stated as an assertion. Everything a
    # resume has to restore -- the trainable weights, both Adam moment
    # tensors, the torch RNG stream that draws the K candidates and the
    # dropout masks, the batch cursor -- shows up here, because getting any
    # one of them wrong makes the two runs diverge. Bit-equality is reachable
    # only because Stage 10.1 keeps the per-molecule forward and the
    # per-molecule sampler; it is the same property that makes it identical to
    # Stage 10, applied across an interrupt instead of across an
    # implementation.
    def _final_state(td, stop_after):
        kw = dict(pairs=list(PAIRS) * 3, save_dir=td, variant="a",
                  num_epochs=1, batch_size=2, k_cand=4, workers=0,
                  ckpt_every=1, auto_resume=True)
        if stop_after:
            real = _InterruptGuard

            class _Stop(_InterruptGuard):
                reads = 0

                @property
                def requested(self):                 # type: ignore[override]
                    type(self).reads += 1
                    return type(self).reads >= stop_after

                @requested.setter
                def requested(self, value):
                    pass

            globals()["_InterruptGuard"] = _Stop
            try:
                run_stage10_1_training(**kw)         # interrupted
            finally:
                globals()["_InterruptGuard"] = real
        run_stage10_1_training(**kw)                 # runs, or resumes, to the end
        return load_checkpoint(td)

    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        straight = _final_state(td_a, stop_after=0)
        resumed  = _final_state(td_b, stop_after=3)

    assert straight["global_step"] == resumed["global_step"] == 6
    a_w, b_w = straight["trainable"], resumed["trainable"]
    assert set(a_w) == set(b_w), "different tensors trained"
    diffs = {k: float((a_w[k] - b_w[k]).abs().max()) for k in a_w}
    worst = max(diffs, key=diffs.get)
    assert diffs[worst] == 0.0, (
        f"resumed weights differ from uninterrupted ones; worst tensor "
        f"{worst} by {diffs[worst]:.3e}")

    a_st, b_st = straight["optimizer"]["state"], resumed["optimizer"]["state"]
    assert set(a_st) == set(b_st), "different optimizer slots"
    for k in a_st:
        for moment in ("exp_avg", "exp_avg_sq"):
            d = float((a_st[k][moment] - b_st[k][moment]).abs().max())
            assert d == 0.0, f"Adam {moment} for param {k} differs by {d:.3e}"
        assert int(a_st[k]["step"]) == int(b_st[k]["step"]) == 6, "step count differs"
    print(f"  [7] a run interrupted at batch 3 and resumed lands on the SAME "
          f"{len(a_w)} weight tensors and the SAME Adam moments as an "
          f"uninterrupted run, exactly   OK")

    print("\nStage 10.1 self-test passed.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        _v, _limit, _seed, _workers, _fresh, _epochs = _parse_args(sys.argv)
        main(variant=_v, max_pairs_per_source=_limit, sample_seed=_seed,
             workers=_workers, fresh=_fresh, num_epochs=_epochs)

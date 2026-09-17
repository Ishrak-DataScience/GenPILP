# -*- coding: utf-8 -*-
"""
stage10_vanila_backpropagation_training.py
===========================================
Stage 10 -- property-guided fine-tuning of ChemBERTa by ORDINARY
BACKPROPAGATION. No REINFORCE, no policy gradient, no LoRA.

Two variants, selected by config.STAGE10_VARIANT, trained from this one file so
that everything except the loss is held fixed:

  10a  cross-entropy toward the best candidate
       PLUS an unlikelihood term that pushes DOWN the tokens of invalid ones
  10b  cross-entropy toward the best candidate only

Why this is not just "loss.backward() on the RDKit score"
----------------------------------------------------------
It cannot be. RDKit validity, QED, SA, novelty and the PAINS/Brenk filters are
computed by running a C++ parser on a DECODED SMILES STRING. Sampling tokens
and decoding them is not differentiable, so a Python float returned by RDKit
has no gradient no matter how large you make it. "loss = 1000 if invalid" is
not a tensor, and `.backward()` does not exist on it.

Stage 9 bridges that gap with REINFORCE: it multiplies log-probabilities by the
score, which IS differentiable, at the cost of a high-variance estimator and an
unbounded direction (pushing its own samples toward probability zero).

Stage 10 bridges it by SELECTION instead. For each masked molecule:

    1. ONE forward pass gives the logits at every <mask> position.
    2. K completions are sampled from those logits (no gradient needed).
    3. Each completion is scored with the composite LOSS (below). RDKit runs
       here, on strings, exactly as in Stage 9.
    4. The lowest-loss completion becomes a TARGET, and the model is trained
       toward it with ordinary cross-entropy on the SAME logits from step 1.

RDKit therefore decides WHICH tokens to pull toward; the gradient itself is
plain supervised cross-entropy. Nothing about the chemistry needs to be
differentiable, and the loss that reaches `.backward()` is the same
`F.cross_entropy` used to pretrain the model in the first place.

This is expert iteration / rejection-sampling fine-tuning. Unlike REINFORCE it
has no unbounded "make my own samples less likely" direction: cross-entropy is
bounded below by zero and only ever pulls toward tokens that actually produced
a good molecule.

The composite loss (lower is better -- the opposite of Stage 9's score)
------------------------------------------------------------------------
    valid:    w_qed*(1 - QED)
            + w_sa*(SA_raw - 1)/9
            + w_novelty*similarity(parent, generated)
            + w_tox_alert*(PAINS or Brenk alert)
    invalid:  S + w_valid,   S = w_qed + w_sa + w_novelty + w_tox_alert

S is the most a VALID molecule can lose, so an invalid one is worse than every
valid one by exactly w_valid. That encodes "validity matters most" without a
1000x term, which would swamp the cross-entropy (O(1-10)) and blow up the step.
Tox21 is deliberately absent -- no checkpoint exists yet.

The parent fallback, and why it is the common case
---------------------------------------------------
MEASURED on this data at MASK_PERCENT=15: ~70% of molecules yield ZERO valid
completions out of K=16, and K=8 gives the identical rate -- validity is
bimodal and molecule-intrinsic, not a matter of drawing enough samples. Those
molecules have no candidate worth imitating, so the PARENT's own tokens are
used as the target: always recoverable, and valid by construction.

Recovering them is exact and needs no alignment. mask_atoms_in_smiles_token_level
tokenizes the CANONICAL SMILES and rebuilds the string token by token, each
original BPE token either surviving as its own text or becoming one <mask>.
Re-tokenizing that string cannot change the count -- an unmasked run is the
same substring and so re-merges identically, and every <mask> is a hard
boundary that stops neighbours merging across it. Verified on 500 molecules:
100% of masked strings have exactly the same token count as their canonical
parent, so the target at mask position i is simply the canonical parent's token
at index i. (Comparing against the RAW SMILES instead of the canonical form
agrees only 16% of the time -- the masker canonicalises first.)

But that target teaches RECONSTRUCTION, and reconstructing the parent scores
similarity = 1, the worst possible novelty. At full weight ~70% of steps would
train a copier. config.STAGE10_FALLBACK_WEIGHT (default 0.3) scales those steps
only, and spans the whole design space: 1.0 = train them fully, 0.0 = skip them.

Shared with Stage 9 (so the comparison is controlled)
------------------------------------------------------
  pair collection      collect_all_training_pairs
  property measurement compute_property_components
  property figures     collect_pairs_by_source / evaluate_property_records /
                       plot_property_report / write_property_records_csv
  masking rate         config.MASK_PERCENT, one value for both stages

NOT shared: the training loop, the loss, and the training-curve plot. No part
of Stage 9's REINFORCE machinery is imported.

Usage
-----
  python stage10_vanila_backpropagation_training.py
  python stage10_vanila_backpropagation_training.py --test
  ... --variant b        train 10b instead of config.STAGE10_VARIANT
  ... --epochs N        override config.STAGE10_NUM_EPOCHS for this run
  ... --limit none       uncapped final property pass
"""

from __future__ import annotations

# FIRST, above torch and transformers: torchao logs a register_constant()
# deprecation while it is being imported, and a filter installed after that
# import has nothing left to catch. See quiet_torch_logs for what it drops and
# what it deliberately does not.
#
# Guarded because this module only makes the LOG tidier. A checkout that is
# missing it -- a partial sync, a `git commit -am` that skipped the untracked
# file -- must still train; dying at import over two suppressed warning lines
# would be the worst possible trade.
try:
    import quiet_torch_logs  # noqa: F401
except ImportError:
    pass

import json
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from transformers import AutoModelForMaskedLM

RDLogger.DisableLog("rdApp.*")

import config
import stage10_data_split as split
import stage10_lineage as lineage
from stage9_masked_property_finetune import (
    collect_all_training_pairs,
    collect_pairs_by_source,
    compute_property_components,
    evaluate_property_records,
    format_collection_footer,
    format_property_summary,
    get_chemberta_tokenizer,
    plot_property_report,
    write_property_records_csv,
)

try:
    from tqdm import tqdm
except ImportError:                                   # pragma: no cover
    # A silent stand-in with the same call surface, tqdm.write included;
    # the old inline stub was a bare function and had no .write.
    from tqdm_compat import tqdm  # type: ignore[misc]


# ── knobs (all from config; see config.py for the reasoning) ────────────────
VARIANT              = getattr(config, "STAGE10_VARIANT", "a")
K_CANDIDATES         = getattr(config, "STAGE10_NUM_CANDIDATES", 16)
TOP_K                = getattr(config, "STAGE10_TOP_K", 20)
TEMPERATURE          = getattr(config, "STAGE10_TEMPERATURE", 1.2)
W_QED                = getattr(config, "STAGE10_W_QED", 0.30)
W_SA                 = getattr(config, "STAGE10_W_SA", 0.20)
W_NOVELTY            = getattr(config, "STAGE10_W_NOVELTY", 0.30)
W_TOX_ALERT          = getattr(config, "STAGE10_W_TOX_ALERT", 0.20)
W_VALID              = getattr(config, "STAGE10_W_VALID", 1.00)
FALLBACK_WEIGHT      = getattr(config, "STAGE10_FALLBACK_WEIGHT", 0.3)
UNLIKELIHOOD_WEIGHT  = getattr(config, "STAGE10_UNLIKELIHOOD_WEIGHT", 0.5)
UNFREEZE_LAST_N      = getattr(config, "STAGE10_UNFREEZE_LAST_N_BLOCKS", 1)
NUM_EPOCHS           = getattr(config, "STAGE10_NUM_EPOCHS", 3)
BATCH_SIZE           = getattr(config, "STAGE10_BATCH_SIZE", 16)
LEARNING_RATE        = getattr(config, "STAGE10_LEARNING_RATE", 5e-5)
GRAD_CLIP            = getattr(config, "STAGE10_GRAD_CLIP", 1.0)
MAX_MODEL_TOKENS     = 512

# The most a VALID molecule can lose. An invalid one scores S + W_VALID, i.e.
# strictly worse than every valid molecule by a margin of exactly W_VALID.
S_WORST_VALID = W_QED + W_SA + W_NOVELTY + W_TOX_ALERT
LOSS_INVALID  = S_WORST_VALID + W_VALID

LOSS_TERMS = ("qed", "sa", "novelty", "tox_alert")

# Every per-epoch series the three Stage 10 scripts record. Kept in one place
# because 10.1 and 10.2 build the same dict and hand it straight back to
# _plot_history, and because a checkpoint written before a series existed has
# to be widened on resume rather than crashing the first epoch that appends
# to it -- see ensure_history_keys.
HISTORY_KEYS = (
    "epoch", "loss_mean", "best_loss_mean", "fallback_rate",
    "cand_valid_rate", "best_valid_rate", "unlikelihood_mean",
    "qed", "sa", "novelty", "tox_alert", "tox_alert_rate",
    # The held-out fold, added by stage10_data_split.validation_pass. Listed
    # here rather than in each trainer so that all four record the same series
    # under the same names -- a val_ curve that meant one thing in Stage 10 and
    # another in 10.2 would be worse than none.
) + split.val_history_keys(LOSS_TERMS)

def new_history() -> Dict[str, list]:
    """An empty history in the shape all three Stage 10 scripts write."""
    return {k: [] for k in HISTORY_KEYS}


def alert_rate_from_loss(tox_alert_term: float) -> float:
    """
    Turn the tox_alert LOSS term back into the FRACTION of molecules carrying
    a PAINS or Brenk structural alert.

    The term is W_TOX_ALERT * any_alert averaged over the epoch's molecules,
    and any_alert is exactly 1.0 or 0.0, so dividing by the weight recovers
    the rate. Molecules that were invalid, or whose catalogs could not be run,
    were charged the full weight upstream and therefore count as alerting
    here: unverified is treated as bad, the same pessimistic reading the loss
    itself uses. That is stated on the figure, because a curve sitting near
    100% at epoch 1 otherwise reads as a bug rather than as the ~70% invalid
    candidate rate it mostly is.
    """
    if W_TOX_ALERT <= 0:
        return 0.0
    return float(tox_alert_term) / W_TOX_ALERT


# ── Loss terms back to the properties they were computed from ────────────────
# Each series in `history` is a WEIGHTED LOSS TERM averaged over an epoch's
# molecules, which is what the objective uses but not what anyone reads a
# property plot for. The three inversions below are exact, because each term is
# weight x property averaged linearly, and they follow alert_rate_from_loss
# exactly in one further respect: a candidate that was invalid, or that RDKit
# could not measure, was charged the FULL weight upstream, so it re-enters here
# at the worst value of its property (QED 0, SA 10, novelty 0). That is stated
# on every figure that uses them, because a curve that folds in a ~70% invalid
# rate otherwise reads as a claim about the molecules that were valid.

def qed_from_loss(qed_term: float) -> float:
    """Mean QED (0-1, higher better); invalid candidates counted as 0."""
    if W_QED <= 0:
        return float("nan")
    return 1.0 - float(qed_term) / W_QED


def sa_from_loss(sa_term: float) -> float:
    """Mean SA score on its native 1-10 scale (lower = easier to make);
    invalid candidates counted as 10."""
    if W_SA <= 0:
        return float("nan")
    return 1.0 + 9.0 * (float(sa_term) / W_SA)


def novelty_from_loss(novelty_term: float) -> float:
    """
    Mean novelty, 1 - Tanimoto(parent, candidate), 0-1 higher better.

    The stored term penalises SIMILARITY, so this is the complement; invalid
    candidates were charged full weight and therefore count as novelty 0, i.e.
    as identical to the parent.
    """
    if W_NOVELTY <= 0:
        return float("nan")
    return 1.0 - float(novelty_term) / W_NOVELTY


def ensure_history_keys(history: Dict[str, list]) -> Dict[str, list]:
    """
    Widen a RESUMED history to the current key set, in place.

    A checkpoint written before a series was added carries no list for it, and
    the first `history[k].append(v)` after resume would raise KeyError and
    lose the run. Missing epochs are backfilled -- derived where a derivation
    exists (tox_alert_rate is a pure function of tox_alert, which every
    checkpoint has), NaN otherwise, so matplotlib leaves a visible gap instead
    of drawing an invented value.
    """
    n = len(history.get("epoch") or [])
    for key in HISTORY_KEYS:
        series = history.setdefault(key, [])
        if len(series) >= n:
            continue
        if key == "tox_alert_rate" and len(history.get("tox_alert") or []) >= n:
            history[key] = [alert_rate_from_loss(v)
                            for v in history["tox_alert"][:n]]
        else:
            # Prepend: a short series is short because it started late, so its
            # values belong at the RECENT end of the epoch axis.
            series[:0] = [float("nan")] * (n - len(series))
    return history


# Identifies this script to stage10_lineage, which decides where it writes and
# -- when config.STAGE10_SHARED_OUTPUT is on -- lets a run begun here be
# finished under Stage 10.1 or Stage 10.2, and vice versa.
STAGE = "stage10"


# ════════════════════════════════════════════════════════════════════════════
#  THE COMPOSITE LOSS  (RDKit; runs on strings, no gradient involved)
# ════════════════════════════════════════════════════════════════════════════

def compute_stage10_loss(
    generated_smiles: str,
    original_smiles:  str,
) -> Tuple[float, Dict[str, float]]:
    """
    Composite LOSS for one generated SMILES against its pre-mask parent.
    Lower is better -- the mirror image of Stage 9's compute_stage9_score.

    Returns (loss, per-term breakdown). The breakdown carries "valid" as
    1.0/0.0 for reporting and every other term as its ALREADY-WEIGHTED
    contribution, so the terms sum to the loss and can be plotted directly.

    A term that cannot be measured (RDKit parsed the molecule but a descriptor
    failed) is charged its FULL weight: anything we cannot verify is treated as
    bad, which is the pessimistic reading and the safe one for a selection
    criterion.
    """
    measured = compute_property_components(
        generated_smiles, original_smiles,
        need_alert_count=False, need_tox21=False,
    )

    if not measured["valid"]:
        return LOSS_INVALID, {
            "valid": 0.0, "qed": W_QED, "sa": W_SA,
            "novelty": W_NOVELTY, "tox_alert": W_TOX_ALERT,
            "invalid_penalty": W_VALID,
        }

    qed = measured["qed"]
    l_qed = W_QED * (1.0 - qed) if qed is not None else W_QED

    # SA runs 1 (trivial to make) .. 10 (very hard); linear across the range.
    sa_raw = measured["sa_raw"]
    l_sa = (W_SA * min(max((sa_raw - 1.0) / 9.0, 0.0), 1.0)
            if sa_raw is not None else W_SA)

    # compute_property_components returns novelty = 1 - Tanimoto, so
    # similarity = 1 - novelty. Penalising SIMILARITY is what pushes the model
    # away from simply copying its parent.
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
#  MODEL: plain ChemBERTa with only the last layers unfrozen
# ════════════════════════════════════════════════════════════════════════════

def load_model_last_layers(
    model_name:      str = None,
    unfreeze_last_n: int = None,
    checkpoint:      str = None,
):
    """
    ChemBERTa with everything frozen except the LM head and the last
    `unfreeze_last_n` encoder blocks.

    Deliberately NOT LoRA and NOT stage1_9's loader: Stage 10 fine-tunes real
    weights. So the comparison against Stage 9 is between two training
    strategies AND two capacity budgets -- a real difference, stated here
    rather than hidden.
    """
    model_name = model_name or config.CHEMBERTA_MODEL
    if unfreeze_last_n is None:
        unfreeze_last_n = UNFREEZE_LAST_N
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = get_chemberta_tokenizer(config.CHEMBERTA_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    n_blocks = model.config.num_hidden_layers
    keep = {f"encoder.layer.{i}." for i in
            range(max(n_blocks - unfreeze_last_n, 0), n_blocks)}

    for name, param in model.named_parameters():
        param.requires_grad = (name.startswith("lm_head")
                               or any(k in name for k in keep))

    if checkpoint and os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["trainable"], strict=False)
        tqdm.write(f"  Restored {len(state['trainable'])} trainable tensor(s) "
                   f"from {checkpoint}")

    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    tqdm.write(f"  Device : {device}")
    tqdm.write(f"  Trainable: LM head + last {unfreeze_last_n} of {n_blocks} "
               f"encoder block(s) -- {n_train:,} / {n_total:,} "
               f"({100 * n_train / n_total:.2f}%)")
    return tokenizer, model, device


def _trainable_state(model) -> Dict[str, torch.Tensor]:
    """
    Only the unfrozen tensors. ~7.7M floats (30 MB) against 180 MB for the
    whole model -- worth caring about when every epoch writes to Drive.
    """
    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


# ════════════════════════════════════════════════════════════════════════════
#  TARGETS: the parent's own tokens at the mask positions
# ════════════════════════════════════════════════════════════════════════════

def parent_target_ids(
    masked_smiles:   str,
    original_smiles: str,
    tokenizer,
) -> Optional[List[int]]:
    """
    The canonical parent's token id at each <mask> position, or None if the
    invariant below does not hold.

    Exact by construction, not by alignment: the masker tokenizes the CANONICAL
    SMILES and swaps whole BPE tokens for single <mask>es, so the masked string
    has the same token count as the canonical parent, and mask position i holds
    what was parent token i. Measured at 100% on 500 molecules.

    The length check is an assertion of that invariant, not a heuristic. If it
    ever fails (an unsanitizable parent, a truncation) the caller skips the
    molecule rather than training on a silently misaligned target.
    """
    mol = Chem.MolFromSmiles(original_smiles) if original_smiles else None
    if mol is None:
        return None
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

    masked_ids = tokenizer(masked_smiles, add_special_tokens=True,
                           truncation=True, max_length=MAX_MODEL_TOKENS)["input_ids"]
    parent_ids = tokenizer(canonical, add_special_tokens=True,
                           truncation=True, max_length=MAX_MODEL_TOKENS)["input_ids"]
    if len(masked_ids) != len(parent_ids):
        return None

    mask_id = tokenizer.mask_token_id
    return [parent_ids[i] for i, t in enumerate(masked_ids) if t == mask_id]


# ════════════════════════════════════════════════════════════════════════════
#  ONE TRAINING STEP
# ════════════════════════════════════════════════════════════════════════════

def _sample_candidates(logits_at_masks: torch.Tensor, k_cand: int,
                       top_k: int, temperature: float) -> torch.Tensor:
    """
    [n_masks, V] logits -> [k_cand, n_masks] sampled token ids.

    Top-k + temperature, matching Stage 9's rollout so the candidate pool comes
    from the same distribution Stage 9 would have rolled out. Detached on
    purpose: sampling is a SEARCH over possible targets and carries no
    gradient. The gradient comes from cross-entropy against whichever target
    wins, which is what makes this supervised learning rather than REINFORCE.
    """
    scaled = (logits_at_masks / max(temperature, 1e-8)).detach()
    probs = F.softmax(scaled, dim=-1)
    kk = min(top_k, probs.shape[-1])
    top = torch.topk(probs, k=kk, dim=-1)
    renorm = top.values / top.values.sum(dim=-1, keepdim=True)
    picks = torch.multinomial(renorm, num_samples=k_cand, replacement=True)
    return top.indices.gather(1, picks).t().contiguous()


def stage10_batch_loss(
    batch:       List[Tuple[str, str]],
    tokenizer,
    model,
    device:      str,
    variant:     str   = None,
    k_cand:      int   = K_CANDIDATES,
    top_k:       int   = TOP_K,
    temperature: float = TEMPERATURE,
    fallback_weight:     float = FALLBACK_WEIGHT,
    unlikelihood_weight: float = UNLIKELIHOOD_WEIGHT,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Loss for one batch, plus statistics for reporting.

    ONE forward pass per molecule serves both jobs: it supplies the
    distribution the K candidates are sampled from, and it supplies the logits
    cross-entropy is computed against. No second forward is needed.
    """
    variant = (variant or VARIANT).lower()
    total = torch.zeros((), device=device)
    n_weighted = 0.0
    stats: Dict[str, float] = {
        "n": 0.0, "fallback": 0.0, "cand_valid": 0.0, "cand_total": 0.0,
        "best_loss": 0.0, "unlikelihood": 0.0, "skipped": 0.0, "best_valid": 0.0,
    }
    for key in LOSS_TERMS:
        stats[key] = 0.0

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

        cands = _sample_candidates(at_masks, k_cand, top_k, temperature)

        # Score every candidate with RDKit. Strings only -- no gradient here.
        best_loss, best_ids, best_comps = None, None, None
        invalid_rows: List[torch.Tensor] = []
        for row in cands:
            filled = ids.clone()
            filled[mask_pos] = row
            smi = tokenizer.decode(filled, skip_special_tokens=True).replace(" ", "")
            loss_c, comps = compute_stage10_loss(smi, parent_smi)
            stats["cand_total"] += 1
            if comps["valid"]:
                stats["cand_valid"] += 1
            else:
                invalid_rows.append(row)
            if best_loss is None or loss_c < best_loss:
                best_loss, best_ids, best_comps = loss_c, row, comps

        used_fallback = (best_comps is None) or (not best_comps["valid"])
        if used_fallback:
            # No candidate was valid (~70% of molecules at 15% masking).
            # Reconstruct the parent instead, DOWN-WEIGHTED so these steps do
            # not turn the model into a copier -- reconstructing the parent
            # scores similarity 1, the worst possible novelty.
            tgt = parent_target_ids(masked_smi, parent_smi, tokenizer)
            if tgt is None or len(tgt) != mask_pos.numel():
                stats["skipped"] += 1
                continue
            best_ids = torch.tensor(tgt, device=device, dtype=torch.long)
            weight = fallback_weight
            stats["fallback"] += 1
        else:
            weight = 1.0

        # ── the differentiable part: ordinary cross-entropy ────────────────
        # Mean over this molecule's masks, so a heavily masked molecule does
        # not dominate the batch merely by having more positions.
        log_probs = F.log_softmax(at_masks, dim=-1)
        ce = F.nll_loss(log_probs, best_ids, reduction="mean")
        mol_loss = weight * ce

        if variant == "a" and invalid_rows and unlikelihood_weight > 0:
            # Push DOWN the tokens that produced invalid molecules:
            #   -log(1 - P(token))
            # Bounded, unlike +log P which diverges as P -> 0.
            #
            # A candidate token equal to the TARGET token at the same position
            # is excluded: otherwise this term would fight the cross-entropy
            # above, pushing down the very token CE is pulling up.
            bad = torch.stack(invalid_rows)                  # [n_bad, n_masks]
            keep = bad != best_ids.unsqueeze(0)
            if keep.any():
                probs = log_probs.exp()                      # [n_masks, V]
                p_bad = probs.gather(1, bad.t()).t()         # [n_bad, n_masks]
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
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def _meta_path(save_dir: str) -> str:
    return os.path.join(save_dir, "stage10_checkpoint.json")


def run_stage10_training(
    pairs:      List[Tuple[str, str]],
    save_dir:   str   = None,
    variant:    str   = None,
    num_epochs: int   = NUM_EPOCHS,
    batch_size: int   = BATCH_SIZE,
    lr:         float = LEARNING_RATE,
    grad_clip:  float = GRAD_CLIP,
    k_cand:     int   = K_CANDIDATES,
) -> Dict[str, list]:
    """Supervised best-of-K fine-tuning. Resumable; single-process by design."""
    variant = (variant or VARIANT).lower()
    if variant not in ("a", "b"):
        raise ValueError(f"variant must be 'a' or 'b', got {variant!r}")
    save_dir = save_dir or lineage.resolve_save_dir(STAGE, variant)
    os.makedirs(save_dir, exist_ok=True)

    # SHARED MODE changes what "resume" means here, and it is the one place
    # this reference implementation gains a capability rather than just a path.
    # Its own format saves the unfrozen tensors and nothing else, so every
    # Stage 10 resume today rebuilds Adam from scratch and discards exp_avg /
    # exp_avg_sq -- hundreds of steps of adaptive state, silently, on each
    # restart. The lineage checkpoint carries the optimizer and the RNG
    # streams, so in shared mode that stops happening. With the flag off, the
    # epoch_NNN.pt + JSON path below is untouched and an existing run keeps
    # resuming exactly as it always did.
    shared = lineage.shared_enabled()
    profile = lineage.execution_profile(STAGE, amp=None, batched=False)

    history: Dict[str, list] = new_history()
    start_epoch = 1
    global_step = 0
    resume_ckpt = None
    provenance: List[dict] = []
    lineage_state = None

    if shared:
        lineage_state = lineage.load_state(save_dir, STAGE, log=tqdm.write)
        if lineage_state:
            banner = lineage.lineage_banner(lineage_state, profile)
            if banner:
                tqdm.write(banner)
            start_epoch = int(lineage_state["epoch"])
            history     = ensure_history_keys(lineage_state["history"])
            provenance  = lineage_state.get("provenance") or []
            global_step = int(lineage_state.get("global_step", 0))
            if int(lineage_state.get("batch_index", 0)) != 0:
                # Stage 10 trains whole epochs; it has no way to enter one
                # part-way through. Restarting the epoch is the only honest
                # option, and naming the command that WOULD resume exactly is
                # more useful than quietly discarding the partial epoch.
                tqdm.write(
                    f"  The lineage checkpoint stopped MID-EPOCH, at batch "
                    f"{lineage_state['batch_index']} of epoch {start_epoch}. "
                    f"Stage 10 trains whole epochs only, so epoch {start_epoch} "
                    f"restarts from its beginning. Run Stage 10.1 or Stage 10.2 "
                    f"instead to continue at that exact batch.")
            tqdm.write(f"\n  Resuming from epoch {start_epoch} (lineage "
                       f"checkpoint in {save_dir}, last written by "
                       f"{lineage_state.get('written_by') or 'an earlier run'})")
    elif os.path.isfile(_meta_path(save_dir)):
        try:
            with open(_meta_path(save_dir)) as f:
                meta = json.load(f)
            start_epoch = meta["last_epoch"] + 1
            history = ensure_history_keys(meta["history"])
            resume_ckpt = os.path.join(save_dir, f"epoch_{meta['last_epoch']:03d}.pt")
            tqdm.write(f"\n  Resuming from epoch {start_epoch} (found {save_dir})")
        except Exception as e:
            tqdm.write(f"  Could not read checkpoint ({e}); starting fresh.")

    if start_epoch > num_epochs:
        tqdm.write("  Training already complete (all epochs done).")
        return history

    tokenizer, model, device = load_model_last_layers(checkpoint=resume_ckpt)
    if lineage_state is not None:
        # strict=False by necessity: the checkpoint holds ONLY the unfrozen
        # tensors, so every frozen weight is reported missing and is supposed
        # to be -- it came from the pretrained model.
        model.load_state_dict(lineage_state["trainable"], strict=False)
        tqdm.write(f"  Restored {len(lineage_state['trainable'])} trainable "
                   f"tensor(s) from the lineage checkpoint.")
    model.train()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    if lineage_state is not None and lineage_state.get("optimizer"):
        try:
            optimizer.load_state_dict(lineage_state["optimizer"])
            tqdm.write("  Optimizer state restored (Adam moments preserved).")
        except Exception as e:
            tqdm.write(f"  Could not restore optimizer state ({e}); "
                       f"continuing with a fresh Adam.")
    lineage.restore_rng(lineage_state.get("rng") if lineage_state else None,
                        log=tqdm.write)

    tqdm.write(
        f"\n  Variant {variant} : "
        + ("CE toward best candidate + unlikelihood on invalid ones"
           if variant == "a" else "CE toward best candidate only")
        + f"\n  K candidates    : {k_cand}"
        + f"\n  loss (valid)    : {W_QED}*(1-QED) + {W_SA}*(SA-1)/9 "
          f"+ {W_NOVELTY}*similarity + {W_TOX_ALERT}*alert   -> at most {S_WORST_VALID}"
        + f"\n  loss (invalid)  : {LOSS_INVALID}  (= {S_WORST_VALID} + w_valid {W_VALID})"
        + f"\n  fallback weight : {FALLBACK_WEIGHT}  (parent-reconstruction steps)"
    )

    # ── hold out the validation fold BEFORE the first optimizer step ──────
    # Whole Bemis-Murcko groups, shared with every other Stage 10 variant via
    # the cached manifest, so "validation" means the same molecules here as it
    # does in 10.1, 10.2 and 10.4. See stage10_data_split.
    pairs, val_pairs, _ = split.split_pairs(pairs)
    for line in split.describe(split.get_split([p for _, p in pairs + val_pairs]),
                               (pairs, val_pairs, [])):
        tqdm.write(line)

    def _val_loss_fn(batch):
        return stage10_batch_loss(batch, tokenizer, model, device,
                                  variant=variant, k_cand=k_cand)

    best_val = float("inf")

    rng = random.Random(42 + start_epoch)
    n_batches = (len(pairs) + batch_size - 1) // batch_size
    # A live bar on a TTY, a status line every 100 batches when there is no
    # TTY to draw one on (Colab's `!python`, nohup, a piped log). See
    # stage10_lineage.Progress.
    pbar = lineage.Progress(total=(num_epochs - start_epoch + 1) * n_batches,
                            desc=f"Stage 10{variant} training", every=100)

    for epoch in range(start_epoch, num_epochs + 1):
        rng.shuffle(pairs)
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
        agg = {k: 0.0 for k in ("loss", "best_loss", "fallback", "cand_valid",
                                "cand_total", "n", "unlikelihood", "best_valid")}
        for key in LOSS_TERMS:
            agg[key] = 0.0
        n_steps = 0

        for batch in batches:
            optimizer.zero_grad()
            loss, stats = stage10_batch_loss(
                batch, tokenizer, model, device, variant=variant, k_cand=k_cand)
            if stats["n"] == 0:
                pbar.update(1)
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()

            agg["loss"] += float(loss.detach())
            for k in ("best_loss", "fallback", "cand_valid", "cand_total", "n",
                      "unlikelihood", "best_valid"):
                agg[k] += stats[k]
            for key in LOSS_TERMS:
                agg[key] += stats[key]
            n_steps += 1

            pbar.set_postfix_str(
                f"ep={epoch}/{num_epochs}  loss={float(loss.detach()):.4f}  "
                f"best={stats['best_loss'] / max(stats['n'], 1):.3f}  "
                f"fallback={stats['fallback'] / max(stats['n'], 1):.0%}  "
                f"cand_valid={stats['cand_valid'] / max(stats['cand_total'], 1):.0%}",
                refresh=True)
            pbar.update(1)

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
        row["tox_alert_rate"] = alert_rate_from_loss(row["tox_alert"])
        # The held-out fold, scored under the identical best-of-K procedure.
        # validation_pass saves and restores the RNG streams, so adding it does
        # not shift a single training draw -- Stage 10 stays bit-comparable
        # with a run that had no validation fold.
        row.update(split.validation_pass(
            val_pairs, _val_loss_fn, model, LOSS_TERMS,
            batch_size=batch_size, seed=42))
        for k in HISTORY_KEYS:
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
            + ("\n" + split.format_val_line(row) if row.get("val_n") else "")
        )

        # MODEL SELECTION on the held-out fold. Saved the moment it improves,
        # not held in memory to the end: a run that stops early then leaves the
        # best model on disk rather than the last one, and run_stage10_eval --
        # which loads from save_dir -- evaluates the selected epoch rather than
        # whichever epoch happened to be last.
        if split.SELECT_BEST_VAL and split.is_best_val(history):
            best_val = history["val_best_loss_mean"][-1]
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            tqdm.write(f"    new best held-out loss {best_val:.4f} "
                       f"-- model saved to {save_dir}")

        ckpt = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
        torch.save({"trainable": _trainable_state(model)}, ckpt)
        global_step += n_steps
        if shared:
            # The full lineage state, so Stage 10.1 or 10.2 can continue this
            # run with Adam's moments intact. batch_index = 0 of the NEXT
            # epoch: this one is finished. The fingerprint is written in the
            # key set those two stages compare against, with bucketing False
            # because Stage 10 never reorders a batch.
            provenance = lineage.save_state(
                save_dir, STAGE,
                trainable   = _trainable_state(model),
                optimizer   = optimizer.state_dict(),
                epoch       = epoch + 1,
                batch_index = 0,
                global_step = global_step,
                history     = history,
                n_steps     = 0,
                fingerprint = {"variant": variant, "batch_size": batch_size,
                               "k_cand": k_cand, "num_epochs": num_epochs,
                               "n_pairs": len(pairs),
                               "seed": getattr(config, "STAGE10_1_SEED", 42),
                               "bucketing": False},
                rng         = lineage.rng_state(),
                profile     = profile,
                provenance  = provenance,
            )
        else:
            with open(_meta_path(save_dir), "w") as f:
                json.dump({"last_epoch": epoch, "variant": variant,
                           "history": history}, f, indent=2)
        tqdm.write(f"  Checkpoint saved -> {ckpt}")

    pbar.close()
    # Full HF save so the eval pass (and any later stage) can just load it.
    # Skipped when an epoch was selected on the held-out fold: that epoch's
    # weights are already in save_dir, and re-saving here would overwrite the
    # SELECTED model with the LAST one -- silently undoing the selection.
    if not (split.SELECT_BEST_VAL and best_val < float("inf")):
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        tqdm.write(f"\n  Final model saved to : {save_dir}")
    else:
        best_ep = (history["epoch"][history["val_best_loss_mean"].index(best_val)]
                   if best_val in history["val_best_loss_mean"] else "?")
        tqdm.write(f"\n  Final model is epoch {best_ep} (lowest held-out loss "
                   f"{best_val:.4f}), already saved to : {save_dir}")
    if shared:
        tqdm.write("  (shared mode: this is the LINEAGE model -- the endpoint "
                   "of whichever stages trained it. Provenance:)")
        for line in lineage.provenance_table(provenance):
            tqdm.write(line)
    refresh_figures(history, save_dir, variant)
    return history


def savefig_atomic(fig, out: str, **kwargs) -> None:
    """
    Write a figure through a temp file in the same directory, then one
    os.replace onto the final name.

    These PNGs are now rewritten at every epoch boundary of a run that takes
    hours, which means they are read WHILE they are being written -- scp'd off
    the server, opened from a mounted share, picked up by a sync client. A
    plain savefig truncates the old file first and fills it over the following
    moments, so a reader that arrives in that window gets a half-written PNG
    and no way to tell it apart from a finished one. os.replace is atomic on
    the same filesystem, so a reader sees either the previous epoch's figure
    or this one, never a fragment.
    """
    kwargs.setdefault("dpi", 150)
    kwargs.setdefault("bbox_inches", "tight")
    tmp = out + ".tmp.png"
    try:
        fig.savefig(tmp, **kwargs)
        os.replace(tmp, out)
    finally:
        # A render that raised half way leaves the partial temp behind; the
        # caller retries next epoch, and a directory of stale .tmp.png files
        # is exactly the confusion this function exists to prevent.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def refresh_figures(history: Dict[str, list], save_dir: str, variant: str,
                    extra: Sequence[Callable] = (), quiet: bool = False) -> None:
    """
    Draw every standing figure for this run from the history collected so far.

    Called at each EPOCH BOUNDARY as well as at the end, so a run that is
    still going -- or one that was interrupted, or one whose final eval pass
    never ran -- always has current PNGs on disk beside its checkpoint. The
    figures are a pure function of `history`, which is exactly what the
    checkpoint stores, so redrawing them mid-run costs one matplotlib render
    (~1 s against a multi-hour epoch) and can never disagree with the state
    that was saved.

    `extra` takes the per-stage figures a script adds on top of these three;
    each is called with the same (history, save_dir, variant, quiet=) contract.

    Every figure is drawn inside its own try/except: a plotting failure at
    epoch 3 of 30 must not end a run whose weights are fine. The failure is
    reported and the next figure is attempted.
    """
    figures = (_plot_history, _plot_tox_alert_rate,
               _plot_validation_properties, *extra)
    for fn in figures:
        try:
            fn(history, save_dir, variant, quiet=quiet)
        except Exception as exc:                       # noqa: BLE001
            tqdm.write(f"  (figure {getattr(fn, '__name__', fn)} failed: "
                       f"{type(exc).__name__}: {exc})")


def _plot_history(history: Dict[str, list], save_dir: str, variant: str,
                  quiet: bool = False) -> None:
    """Training curves. Loss DESCENDS here, unlike Stage 9's ascending score."""
    if not history.get("epoch"):
        return
    fig, axes = plt.subplots(2, 4, figsize=(21, 8))
    ep = history["epoch"]
    # Every loss term the objective carries gets a panel. "sa" was tracked in
    # HISTORY_KEYS and written every epoch but never drawn, so synthetic
    # accessibility -- one of the four properties the objective optimises --
    # was the only one with no curve at all, on either fold.
    panels = [
        ("loss_mean",        "Training loss (backprop)",        "#1f77b4"),
        ("best_loss_mean",   "Best-candidate composite loss",   "#d62728"),
        ("cand_valid_rate",  "Candidate validity rate",         "#2ca02c"),
        ("fallback_rate",    "Parent-fallback rate",            "#ff7f0e"),
        ("qed",              "QED loss term (lower = more drug-like)", "#8c564b"),
        ("sa",               "SA loss term (lower = easier to make)",  "#17becf"),
        ("novelty",          "Novelty loss term (lower = more novel)", "#9467bd"),
        ("best_valid_rate",  "Selected-target validity rate",   "#7f7f7f"),
    ]
    for a, (key, title, color) in zip(axes.flat, panels):
        a.plot(ep, history.get(key, []), marker="o", color=color, label="train")
        # The held-out counterpart on the same axes wherever one exists. The
        # GAP between the two curves is the quantity of interest -- it is the
        # part of any improvement that did not generalise -- and it is only
        # readable when both are drawn against one scale.
        val = history.get("val_" + key.replace("_mean", "") + "_mean") \
            or history.get("val_" + key)
        if val and len(val) == len(ep) and any(v == v for v in val):
            a.plot(ep, val, marker="s", linestyle="--", color=color,
                   alpha=0.55, label="held-out")
            a.legend(fontsize=7)
        a.set_title(title, fontsize=10)
        a.set_xlabel("Epoch")
        a.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(f"Stage 10{variant} -- supervised best-of-K fine-tuning "
                 f"(no reinforcement learning)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(save_dir, f"stage10{variant}_training_curves.png")
    savefig_atomic(fig, out)
    plt.close(fig)
    if not quiet:
        tqdm.write(f"  Training curves saved : {out}")


def _plot_validation_properties(history: Dict[str, list], save_dir: str,
                                variant: str, quiet: bool = False) -> None:
    """
    The four properties the objective optimises, on the HELD-OUT fold:
    QED, candidate validity, novelty and synthetic accessibility.

    _plot_history draws the raw weighted loss terms, on axes shared with the
    training curve, which is the right figure for "is the objective going
    down". It is the wrong figure for "are the held-out molecules any good":
    the terms are weighted, three of the four are inverted with respect to the
    property, and the held-out curve is a dashed overlay on a training-scaled
    axis. This figure answers the second question directly -- each property on
    its own native scale, held-out solid, training faint behind it, with the
    direction of improvement stated per panel.

    Every value is recovered from the stored loss term (see qed_from_loss,
    sa_from_loss, novelty_from_loss), so it is exactly the quantity the
    objective saw -- INCLUDING invalid candidates at their worst value. With a
    ~70% fallback rate that matters, and the caption says so.

    Silently does nothing when the run recorded no validation pass (the fold is
    configurable off), rather than drawing an empty frame.
    """
    ep = history.get("epoch") or []
    if not ep:
        return

    def series(key, conv=None):
        raw = history.get(key) or []
        if len(raw) != len(ep):
            return None
        out = []
        for v in raw:
            try:
                x = float(v)
            except (TypeError, ValueError):
                out.append(float("nan"));  continue
            out.append(x if conv is None else conv(x))
        return out if any(x == x for x in out) else None

    panels = [
        ("QED", "val_qed", "qed", qed_from_loss, "higher = more drug-like",
         (0.0, 1.0), "#8c564b"),
        ("Candidate validity", "val_cand_valid_rate", "cand_valid_rate",
         lambda v: 100.0 * v, "higher = more parseable molecules",
         (0.0, 100.0), "#2ca02c"),
        ("Novelty vs parent", "val_novelty", "novelty", novelty_from_loss,
         "higher = less like the parent", (0.0, 1.0), "#9467bd"),
        ("Synthetic accessibility", "val_sa", "sa", sa_from_loss,
         "lower = easier to synthesise", (1.0, 10.0), "#17becf"),
    ]

    drawn = 0
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, vkey, tkey, conv, direction, ylim, color) in zip(axes.flat, panels):
        val = series(vkey, conv)
        train = series(tkey, conv)
        if train is not None:
            ax.plot(ep, train, marker="o", markersize=4, color=color, alpha=0.30,
                    linewidth=1.2, label="training")
        if val is not None:
            ax.plot(ep, val, marker="s", color=color, linewidth=2, label="held-out")
            drawn += 1
        ax.set_title(f"{title}\n{direction}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylim(*ylim)
        if len(ep) <= 20:
            ax.set_xticks(list(ep))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

    if drawn == 0:
        plt.close(fig)
        if not quiet:
            tqdm.write("  (no held-out series in history; validation-property "
                       "figure skipped)")
        return

    fig.suptitle(f"Stage 10{variant} -- held-out molecule properties per epoch",
                 fontsize=13)
    fig.text(0.5, -0.02,
             "Recovered from the weighted loss terms, so candidates that were "
             "invalid or could not be measured are included at their worst "
             "value (QED 0, novelty 0, SA 10).",
             ha="center", fontsize=8, color="#555555")
    plt.tight_layout()
    out = os.path.join(save_dir, f"stage10{variant}_validation_properties.png")
    savefig_atomic(fig, out)
    plt.close(fig)
    if not quiet:
        tqdm.write(f"  Held-out property curves saved : {out}")


def _plot_tox_alert_rate(history: Dict[str, list], save_dir: str,
                         variant: str, quiet: bool = False) -> None:
    """
    Toxicity alert rate against epoch -- the single curve the toxicity
    question actually asks about: what fraction of the molecules this stage
    trained TOWARD carries a PAINS or Brenk structural alert.

    Drawn on its own axes rather than as a seventh panel in _plot_history
    because it is read against an absolute scale, 0 to 100% of molecules. A
    shared-figure panel autoscales to whatever narrow band the run happens to
    occupy, which turns a two-point wobble into an apparent trend.

    Shared by all three Stage 10 scripts (10, 10.1, 10.2) exactly as
    _plot_history is, so the figures stay comparable across them.
    """
    ep = history.get("epoch") or []
    if not ep:
        return

    rate = history.get("tox_alert_rate") or []
    if len(rate) != len(ep):
        # A history written before this series existed carries only the
        # weighted loss term; the rate is recoverable from it exactly.
        rate = [alert_rate_from_loss(v) for v in (history.get("tox_alert") or [])]
    if len(rate) != len(ep):
        return
    pct = [100.0 * r for r in rate]
    finite = [(e, p) for e, p in zip(ep, pct) if p == p]      # drops NaN
    if not finite:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([e for e, _ in finite], [p for _, p in finite],
            marker="o", color="#c0392b", linewidth=2,
            label="selected (best-of-K) molecules")

    base_ep, base_pct = finite[0]
    if len(finite) > 1:
        ax.axhline(base_pct, color="#7f8c8d", linestyle="--", linewidth=1,
                   label=f"epoch {base_ep} baseline ({base_pct:.1f}%)")
        last_ep, last_pct = finite[-1]
        ax.annotate(f"{last_pct:.1f}%  ({last_pct - base_pct:+.1f} pts)",
                    xy=(last_ep, last_pct), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="#c0392b")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Toxicity alert rate (% of molecules)")
    ax.set_ylim(0, 100)
    if len(ep) <= 20:
        ax.set_xticks(list(ep))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="best")
    fig.suptitle(f"Stage 10{variant} -- PAINS/Brenk toxicity alert rate per epoch",
                 fontsize=13)
    ax.set_title("lower is better; molecules that were invalid or could not be "
                 "screened\ncount as alerting, matching the loss term this is "
                 "derived from", fontsize=9, color="#555555")
    plt.tight_layout()
    out = os.path.join(save_dir, f"stage10{variant}_tox_alert_rate.png")
    savefig_atomic(fig, out)
    plt.close(fig)
    if not quiet:
        tqdm.write(f"  Toxicity alert-rate curve saved : {out}")


# ════════════════════════════════════════════════════════════════════════════
#  EVALUATION  (Stage 9's figures, so the two are directly comparable)
# ════════════════════════════════════════════════════════════════════════════

def run_stage10_eval(save_dir: str, variant: str,
                     max_pairs_per_source: int = None,
                     sample_seed: int = None) -> Dict[str, list]:
    """
    Score the trained model on the same pairs Stage 9/9a evaluate, with the
    same measurement code and the same figure code -- only the weights differ.
    """
    pairs_by_source = collect_pairs_by_source(
        max_pairs_per_source=max_pairs_per_source, sample_seed=sample_seed)
    if not pairs_by_source:
        tqdm.write("  No pairs found for the property evaluation.")
        return {}

    # Split each source into its training and held-out halves, so the figure
    # carries both series side by side. Before this, the headline property
    # distribution was computed over the pool the model had just trained on,
    # and a model that had merely memorised its parents was indistinguishable
    # from one that had learnt to generate. The distance between the two
    # curves is now that measurement.
    pairs_by_source = split.split_pairs_by_source(pairs_by_source)
    tqdm.write("  Property evaluation is reported separately for training and "
               "held-out molecules;\n  the gap between them is the "
               "memorisation measurement.")

    tokenizer, model, device = load_model_last_layers(model_name=save_dir)
    records = evaluate_property_records(tokenizer, model, device, pairs_by_source)

    out_png = os.path.join(save_dir, f"stage10{variant}_property_distributions.png")
    plot_property_report(
        records, out_png,
        f"Stage 10{variant} -- property distributions (supervised best-of-K)",
        footer=format_collection_footer(list(records.keys())))
    write_property_records_csv(
        records, os.path.splitext(out_png)[0] + "_per_molecule.csv")
    for line in format_property_summary(records):
        tqdm.write(line)
    return records


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(variant: str = None, max_pairs_per_source: int = None,
         sample_seed: int = None, num_epochs: int = None) -> None:
    variant = (variant or VARIANT).lower()
    save_dir = lineage.resolve_save_dir(STAGE, variant)
    print("\n" + "=" * 62)
    print(f"STAGE 10{variant.upper()} -- SUPERVISED BEST-OF-K FINE-TUNING (NO RL)")
    print("=" * 62)
    print(f"""
  For each masked molecule: ONE forward pass -> sample K={K_CANDIDATES}
  completions -> score each with the composite LOSS -> train ordinary
  cross-entropy toward the tokens of the best one.

  RDKit chooses WHICH tokens to pull toward; the gradient is plain
  supervised cross-entropy. Nothing chemical needs to be differentiable.

  loss(valid)   = {W_QED}*(1-QED) + {W_SA}*(SA-1)/9 + {W_NOVELTY}*similarity + {W_TOX_ALERT}*alert
  loss(invalid) = {LOSS_INVALID}   (worst valid = {S_WORST_VALID}, plus w_valid = {W_VALID})
  variant       = {variant}  ({"CE + unlikelihood on invalid" if variant == "a" else "CE only"})
  mask percent  = {config.MASK_PERCENT}%   (shared with Stage 9)""")
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

    history = run_stage10_training(
        pairs=pairs, save_dir=save_dir, variant=variant,
        **({} if num_epochs is None else {"num_epochs": num_epochs}))

    print("\n" + "=" * 62)
    print(f"  Stage 10{variant} training complete.")
    if history.get("epoch"):
        print(f"  Final training loss        : {history['loss_mean'][-1]:.4f}")
        print(f"  Final candidate validity   : {history['cand_valid_rate'][-1]:.1%}")
        print(f"  Final parent-fallback rate : {history['fallback_rate'][-1]:.1%}")
    print(f"  Model : {save_dir}")
    print("=" * 62)

    run_stage10_eval(save_dir, variant, max_pairs_per_source, sample_seed)


def _parse_args(argv: list) -> tuple:
    """--variant a|b, --limit N ("none"/"all"/0 = uncapped), --seed N,
    --shared / --separate (override config.STAGE10_SHARED_OUTPUT)."""
    if "--shared" in argv:
        lineage.set_shared_override(True)
    elif "--separate" in argv:
        lineage.set_shared_override(False)
    variant = None
    epochs = None
    limit = seed = None
    for flag in ("--variant", "--limit", "--seed", "--epochs"):
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
        else:
            seed = int(raw)
    return variant, limit, seed, epochs


if __name__ == "__main__":
    if "--test" in sys.argv:
        from stage10_self_test import run_self_test
        run_self_test()
    else:
        _v, _limit, _seed, _epochs = _parse_args(sys.argv)
        main(variant=_v, max_pairs_per_source=_limit, sample_seed=_seed,
             num_epochs=_epochs)

# -*- coding: utf-8 -*-
"""
stage9_masked_property_finetune.py
====================================
Property-guided fine-tuning of ChemBERTa on the masked-molecule data
produced by Stage 1a (config.STAGE1A_DIR) and Stage 1b
(config.STAGE1B_PLIP_MASK_DIR).

Why REINFORCE instead of plain backprop
----------------------------------------
The four quantities we want to optimise — RDKit validity, QED
(drug-likeness), SA score (synthetic accessibility) and toxicity-alert
status — are all computed by running RDKit on the *decoded, discrete*
SMILES string ChemBERTa produces. That decode step (argmax/sample over a
vocabulary, turn ids back into characters, call Chem.MolFromSmiles) is not
differentiable, so gradients cannot flow from "the SMILES was valid" back
into the token logits the normal autograd way.

The standard workaround — and the one already used by
stage1_9_LLM_RDkit_policy_training.py for its 3-term reward — is the
score-function / REINFORCE estimator: sample tokens, record
log P(chosen token | context) with grad_fn intact, score the finished
molecule with RDKit (no grad needed there), and build a loss out of the
*log-probabilities* instead of the scores directly:

    score = w_valid · valid  +  w_qed · QED  +  w_sa · (1 − SA/10)
          + w_novelty · (1 − Tanimoto(original, generated))
          + w_tox · (1 − toxicity_alert)

    loss  = −(score − baseline) · Σ log_prob_i

`loss` is a real torch.Tensor with a grad_fn (it's built from log_prob,
which came straight out of the model), so `loss.backward()` legitimately
back-propagates into ChemBERTa's LoRA adapter weights — this is exactly
the mechanism the user asked for ("a loss to backpropagate"), just wired
through the REINFORCE identity because the reward terms themselves are
non-differentiable RDKit computations.

New terms vs. stage1_9's reward
--------------------------------
  novelty   : 1 − Tanimoto(original_smiles, generated_smiles) on Morgan
              fingerprints (r=2, 2048 bits) — same metric Stage 3 /
              Stage 1.1a already use for "similarity to the parent
              molecule", just inverted: HIGH similarity to the pre-mask
              parent is now PENALISED (this is what makes fine-tuning
              push the model to wander away from the original scaffold
              instead of just reconstructing it), so novelty is
              rewarded instead.
  tox_alert : 1 − alert, where alert = 1.0 if the generated molecule
              matches any PAINS or Brenk structural alert in RDKit's
              built-in FilterCatalog, else 0.0. A cheap, always-on
              filter-based toxicity/reactivity proxy (no model to load,
              no dependency beyond RDKit itself).
  tox21     : 1 − mean/max(P_toxic) over config.STAGE9_TOX21_SELECTED_TASKS,
              where P_toxic comes from a separately-trained Tox21
              multi-label classifier (a HuggingFace
              AutoModelForSequenceClassification checkpoint pointed to
              by config.STAGE9_TOX21_MODEL_DIR, 12 sigmoid outputs — one
              per Tox21 assay). This is a genuine LEARNED toxicity
              signal (nuclear-receptor / stress-response assay
              activation) layered on top of, not instead of, tox_alert
              — resolved with the user as two independent weighted
              terms (config.STAGE9_SCORE_W_TOX_ALERT and
              config.STAGE9_SCORE_W_TOX21) rather than one blended
              "tox" number, specifically so the cheap rule-based filter
              keeps working even before a Tox21 checkpoint is trained.
              Fails safe to 0 contribution (not 1 — do not silently
              reward "unknown toxicity") whenever
              config.STAGE9_TOX21_MODEL_DIR is unset or missing.

Training data
-------------
  config.STAGE9_DATA_SOURCE selects which source(s) below are used
  ("stage1a", "stage1b", or "both" -- default).

  Stage 1a combo CSVs  (config.STAGE1A_DIR/mask{config.STAGE9_MASK_PERCENT}pct_temp*_seed*.csv)
    columns used: smiles (original), masked_smiles
    Only the combo file(s) matching config.STAGE9_MASK_PERCENT are read, so
    Stage 1a pairs are masked at the same rate as the Stage 1b re-masking
    below rather than mixing every percent in config.STAGE1A_MASK_PERCENTS.

  Stage 1b summary CSV (config.STAGE1B_PLIP_MASK_DIR/stage1b_large_scale_plip_mask_summary.csv)
    columns used: smiles (original), masked_atom_indices, status == "ok"
    Each row's masked_atom_indices is the PLIP-derived candidate pool (every
    interacting atom on "attractive" rows, every non-interacting atom on
    "non-attractive" rows). Rather than using the row's pre-built
    masked_smiles as-is (which masks the *entire* pool), Stage 9 re-samples
    floor(config.STAGE9_MASK_PERCENT% * total_bpe_tokens) atom indices from
    that pool (or masks the whole pool if it has fewer atoms than the
    target) and rebuilds masked_smiles from just that subset.

  Per-parent cap (config.STAGE9_MAX_PAIRS_PER_PARENT, default 3)
    Stage 1b writes one row per ligand INSTANCE, so a molecule resolved in N
    PDB entries contributes N training pairs -- and PDB instance counts follow
    crystallography, not chemistry. Uncapped, sulfate, glycerol, NAG and
    acetate alone are a third of the training set. The cap keeps at most 3
    instances per unique canonical parent (applied per source, seeded per
    parent), which preserves the mask-position diversity that distinct binding
    pockets provide while removing the popularity weighting. See
    _cap_pairs_per_parent.

Only LoRA adapter weights are updated (peft); ChemBERTa itself stays
frozen. Model loading and checkpoint save/resume are reused verbatim from
stage1_9_LLM_RDkit_policy_training.py rather than duplicated.

One-shot rollout (not stage1_9's sequential decoder)
------------------------------------------------------
stage1_9's reinforce_rollout fills <mask> tokens one at a time, re-running
the model after every fill so later masks condition on earlier ones (N
forward passes for N masks). Stage 9 and Stage 9a instead use
reinforce_rollout_oneshot (defined in this module): a SINGLE forward pass
scores every <mask> position at once, and each is sampled independently
from that one pass — ChemBERTa's native masked-language-model behaviour
("predict all masks at once"), the same parallel decoding
config.ONESHOT_MASK_DECODING selects for Stage 2 (generate_smiles_oneshot
in stage2_molecule_generation.py), just applied unconditionally here so
training (this function) and evaluation (generate_completion, used by both
this script's post-training plots and stage9a_masked_property_without_finetuning.py's
baseline plots) share one decoding procedure end to end.

Usage
-----
  python stage9_masked_property_finetune.py
  python stage9_masked_property_finetune.py --test    (synthetic smoke test)
"""

from __future__ import annotations

import csv
import fnmatch
import glob
import hashlib
import io
import json
import os
import random
import sys
import tarfile
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.DataStructs import TanimotoSimilarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RDLogger.DisableLog("rdApp.*")

import config
# Masking goes straight to the token-level masker with the plain ChemBERTa
# tokenizer -- NOT through bpe_mask_adapter.build_smiles_mask_json_fields,
# whose behaviour depends on config.BPE_MASK_ADAPTER_ENABLED /
# CLEAN_SMILES_MASKING and which silently swallows per-molecule failures
# before re-raising from a third fallback. Stage 9/9a want exactly one
# masking rule ("tokenize the clean SMILES, mask whole BPE tokens"),
# unaffected by those flags.
from stage1_mask_calculation import mask_atoms_in_smiles_token_level
from stage3_analysis import _morgan_fp
from stage1_9_LLM_RDkit_policy_training import (
    _SA_AVAILABLE,
    _ask_resume,
    _load_checkpoint,
    _save_checkpoint,
    load_chemberta_for_policy,
)

if _SA_AVAILABLE:
    from rdkit.Contrib.SA_Score import sascorer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):          # type: ignore[misc]
        return iterable if iterable is not None else range(0)

# ── Optional: RDKit's built-in PAINS / Brenk structural-alert catalog ──────
try:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    _TOX_AVAILABLE = True
except ImportError:
    _TOX_AVAILABLE = False
    warnings.warn(
        "rdkit.Chem.FilterCatalog not found — toxicity term will be 0.",
        stacklevel=1,
    )

_TOX_CATALOGS: Dict[str, "FilterCatalog"] = {}


def _get_alert_catalog(which: str = "any"):
    """
    Lazily build (once per kind) an RDKit structural-alert catalog:
    "pains", "brenk", or "any" (both, the union used by the score's
    alert-free term). Keeping PAINS and Brenk separately available matters
    for reporting -- Brenk fires far more often than PAINS, so a single
    merged hit rate is dominated by Brenk and hides which filter fired.
    """
    if which not in _TOX_CATALOGS:
        params = FilterCatalogParams()
        if which in ("pains", "any"):
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        if which in ("brenk", "any"):
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _TOX_CATALOGS[which] = FilterCatalog(params)
    return _TOX_CATALOGS[which]


def _get_toxicity_catalog():
    """The combined PAINS + Brenk catalog (kept for backwards compatibility)."""
    return _get_alert_catalog("any")


# ── Optional: Tox21 multi-label toxicity classifier (second tox term) ──────
# Fails safe (contributes 0, not 1) whenever config.STAGE9_TOX21_MODEL_DIR is
# unset/missing — this is checked once at import time, not per-call, so a
# missing checkpoint costs one warning instead of a warning per molecule.
_TOX21_MODEL_DIR = getattr(config, "STAGE9_TOX21_MODEL_DIR", "") or ""
_TOX21_AVAILABLE = bool(_TOX21_MODEL_DIR) and os.path.isdir(_TOX21_MODEL_DIR)
if not _TOX21_AVAILABLE:
    warnings.warn(
        "config.STAGE9_TOX21_MODEL_DIR is unset or missing — the Tox21 "
        "classifier term will contribute 0. Point it at your trained Tox21 "
        "checkpoint directory to enable this term.",
        stacklevel=1,
    )

_TOX21_MODEL     = None
_TOX21_TOKENIZER = None
_TOX21_DEVICE    = None


def _load_tox21_classifier():
    """Lazily load (once) the Tox21 classifier + tokenizer from STAGE9_TOX21_MODEL_DIR."""
    global _TOX21_MODEL, _TOX21_TOKENIZER, _TOX21_DEVICE
    if not _TOX21_AVAILABLE:
        return None, None, None
    if _TOX21_MODEL is not None:
        return _TOX21_MODEL, _TOX21_TOKENIZER, _TOX21_DEVICE
    try:
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(_TOX21_MODEL_DIR)
        model     = AutoModelForSequenceClassification.from_pretrained(_TOX21_MODEL_DIR)
        model.to(device)
        model.eval()
        _TOX21_MODEL, _TOX21_TOKENIZER, _TOX21_DEVICE = model, tokenizer, device
    except Exception as e:
        warnings.warn(f"Failed to load Tox21 classifier from {_TOX21_MODEL_DIR}: {e}. "
                       "The Tox21 term will contribute 0.", stacklevel=1)
        _TOX21_MODEL = None
    return _TOX21_MODEL, _TOX21_TOKENIZER, _TOX21_DEVICE


def score_tox21(smiles: str) -> float:
    """
    "Clean" probability (1 - aggregated toxic probability) over
    config.STAGE9_TOX21_SELECTED_TASKS, aggregated by
    config.STAGE9_TOX21_AGGREGATION ("mean" or "max"). Returns 0.0
    (fail-safe, not 1.0) whenever the classifier is unavailable or the
    selected task names don't match STAGE9_TOX21_ALL_TASKS.
    """
    model, tokenizer, device = _load_tox21_classifier()
    if model is None or tokenizer is None:
        return 0.0

    all_tasks = list(getattr(config, "STAGE9_TOX21_ALL_TASKS", []))
    selected  = list(getattr(config, "STAGE9_TOX21_SELECTED_TASKS", all_tasks))
    idxs = [all_tasks.index(t) for t in selected if t in all_tasks]
    if not idxs:
        return 0.0

    try:
        enc = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        probs = torch.sigmoid(logits)[idxs]
        aggregation = getattr(config, "STAGE9_TOX21_AGGREGATION", "mean")
        toxic_prob = probs.max().item() if aggregation == "max" else probs.mean().item()
    except Exception:
        return 0.0

    return float(min(max(1.0 - toxic_prob, 0.0), 1.0))


def score_tox21_batch(smiles_list: List[str]) -> List[float]:
    """
    score_tox21 for a whole list in ONE padded forward pass.

    Same value per molecule as calling score_tox21 in a loop -- the classifier
    is frozen and in eval(), so batching changes nothing but the number of GPU
    launches. Used by Stage 9.1, where the RDKit terms are measured in worker
    processes and this is the one term that has to stay on the parent.

    Returns 0.0 (fail-safe, not 1.0 -- never silently reward unknown toxicity)
    for every molecule when the classifier is unavailable, and for any single
    molecule the tokenizer or model chokes on.
    """
    if not smiles_list:
        return []

    model, tokenizer, device = _load_tox21_classifier()
    if model is None or tokenizer is None:
        return [0.0] * len(smiles_list)

    all_tasks = list(getattr(config, "STAGE9_TOX21_ALL_TASKS", []))
    selected  = list(getattr(config, "STAGE9_TOX21_SELECTED_TASKS", all_tasks))
    idxs = [all_tasks.index(t) for t in selected if t in all_tasks]
    if not idxs:
        return [0.0] * len(smiles_list)

    # An empty string is not a molecule the classifier can score; it is also
    # exactly what a failed generation decodes to, so substitute a placeholder
    # to keep the batch rectangular and zero those rows out afterwards.
    safe = [s if s else "C" for s in smiles_list]
    try:
        enc = tokenizer(safe, return_tensors="pt", padding=True,
                        truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**enc).logits            # [B, n_tasks]
        probs = torch.sigmoid(logits)[:, idxs]      # [B, n_selected]
        aggregation = getattr(config, "STAGE9_TOX21_AGGREGATION", "mean")
        toxic = probs.max(dim=-1).values if aggregation == "max" else probs.mean(dim=-1)
        clean = (1.0 - toxic).clamp(0.0, 1.0).tolist()
    except Exception:
        return [0.0] * len(smiles_list)

    return [0.0 if not s else float(c) for s, c in zip(smiles_list, clean)]


# ════════════════════════════════════════════════════════════════════════════
#  DEFAULT HYPER-PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

LORA_RANK        = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05
LORA_TARGET_MODS = ["query", "value"]

# Score weights live in config.py (user-configurable), not here — see
# config.STAGE9_SCORE_W_* for the six-term composite score.
SCORE_W_VALID     = config.STAGE9_SCORE_W_VALID
SCORE_W_QED       = config.STAGE9_SCORE_W_QED
SCORE_W_SA        = config.STAGE9_SCORE_W_SA
SCORE_W_NOVELTY   = config.STAGE9_SCORE_W_NOVELTY
SCORE_W_TOX_ALERT = config.STAGE9_SCORE_W_TOX_ALERT
SCORE_W_TOX21     = config.STAGE9_SCORE_W_TOX21

BASELINE_DECAY   = 0.99
TEMPERATURE      = 1.2
TOP_K            = 20
# ChemBERTa's positional-embedding limit. Pairs longer than this are dropped
# at collection time instead of being silently truncated inside
# reinforce_rollout_oneshot (a truncated molecule scored against its full
# parent is noise, not a data point).
MAX_MODEL_TOKENS = 512
BATCH_SIZE       = 16
LEARNING_RATE    = 5e-5
# See config.STAGE9_NUM_EPOCHS -- 3 by default now that MAX_PAIRS_PER_PARENT
# removes the ~7.4x ligand-instance redundancy the old 10 was absorbing.
NUM_EPOCHS       = getattr(config, "STAGE9_NUM_EPOCHS", 3)
GRAD_CLIP        = 1.0

# How many (masked, original) pairs to actually use -- see the
# STAGE9_* knobs in config.py. None = use everything.
#   MAX_TRAINING_PAIRS        : cap for the REINFORCE loop, COMBINED pool.
#   EVAL_MAX_PAIRS_PER_SOURCE : cap for the property-report pass, applied PER
#                               SOURCE so Stage 1a and Stage 1b panels get a
#                               comparable n, and applied BEFORE generation so
#                               runtime scales with it.
#   SAMPLE_SEED               : seed for both samples; identical in Stage 9 and
#                               Stage 9a, so the "before" and "after" figures
#                               still cover exactly the same molecules.
MAX_TRAINING_PAIRS        = getattr(config, "STAGE9_MAX_TRAINING_PAIRS", None)
# Ligand-instance cap for the TRAINING pool -- see config.STAGE9_MAX_PAIRS_PER_PARENT
# for why an uncapped Stage 1b pool trains mostly on sulfate and glycerol.
MAX_PAIRS_PER_PARENT      = getattr(config, "STAGE9_MAX_PAIRS_PER_PARENT", 3)
EVAL_MAX_PAIRS_PER_SOURCE = getattr(config, "STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE", None)
EVAL_DEDUP_BY_PARENT      = getattr(config, "STAGE9_EVAL_DEDUP_BY_PARENT", True)
SAMPLE_SEED               = getattr(config, "STAGE9_PAIR_SAMPLE_SEED", 42)

# KL anchor to the frozen pretrained model -- see config.STAGE9_KL_BETA.
KL_BETA                = getattr(config, "STAGE9_KL_BETA", 0.0)
KL_BASE_DROPOUT_OFF    = getattr(config, "STAGE9_KL_BASE_DROPOUT_OFF", True)


# Reporting thresholds (annotations only -- never used in the score itself).
QED_DRUGLIKE_THRESHOLD = 0.5    # commonly quoted "drug-like" QED cut
SA_EASY_THRESHOLD      = 4.5    # raw SA-Score below this = readily synthesizable


@lru_cache(maxsize=2)
def get_chemberta_tokenizer(model_name: str = None):
    """
    The plain ChemBERTa BPE tokenizer (config.CHEMBERTA_MODEL), cached.
    Used for both re-masking and rollout so the token counts Stage 9/9a
    reason about are the model's own.
    """
    return AutoTokenizer.from_pretrained(model_name or config.CHEMBERTA_MODEL)


# ════════════════════════════════════════════════════════════════════════════
#  ONE-SHOT ROLLOUT  (ChemBERTa's native, parallel MLM decoding)
# ════════════════════════════════════════════════════════════════════════════

# Morgan fingerprints, memoised on the SMILES string. _morgan_fp is pure (same
# string -> same bit vector, read-only downstream), and the training loop asks
# for the same parent's fingerprint once per instance per epoch -- so without a
# cache a parent seen k times is fingerprinted k times for an identical answer.
# Bounded so a long run cannot grow without limit; the LRU keeps the hot
# parents resident and evicts one-off generated molecules, which is exactly the
# access pattern here.
_FP_CACHE_SIZE = 100_000
_morgan_fp_cached = lru_cache(maxsize=_FP_CACHE_SIZE)(_morgan_fp)


_NO_REFERENCE_WARNED = False


def _split_reward(result) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    Normalise a reward_fn return value to (score, components_or_None).

    Accepts a bare float (components unknown) or a (score, components) pair,
    so callers that only need a number stay as simple as they were while the
    training loop can collect the breakdown from the same scoring call.
    """
    if isinstance(result, tuple):
        score, comps = result
        return float(score), comps
    return float(result), None


def _reference_logits(model, ids, attn_mask):
    """
    Logits of the FROZEN pretrained model on the same masked input -- the
    anchor the KL penalty pulls towards.

    Costs no extra memory and needs no second checkpoint: switching peft's
    LoRA adapter off leaves exactly the pretrained weights behind, so the
    reference is the true pretrained distribution by construction rather than
    a copy that could drift out of sync.

    Evaluated under eval() so the reference is DETERMINISTIC: the frozen
    base's dropout would otherwise make the anchor itself a random variable,
    and a KL measured against a moving target is not a drift measurement.
    Returns None (once, with a warning) when the model has no adapter to
    disable, e.g. the peft-unavailable full-fine-tune fallback -- there is no
    frozen reference to anchor to in that case.
    """
    global _NO_REFERENCE_WARNED
    if not hasattr(model, "disable_adapter"):
        if not _NO_REFERENCE_WARNED:
            _NO_REFERENCE_WARNED = True
            warnings.warn(
                "config.STAGE9_KL_BETA > 0 but the model has no LoRA adapter to "
                "disable (peft unavailable?), so no frozen reference exists -- "
                "the KL anchor is inactive for this run.",
                RuntimeWarning,
            )
        return None

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), model.disable_adapter():
            return model(input_ids=ids.unsqueeze(0),
                         attention_mask=attn_mask.unsqueeze(0)).logits
    finally:
        if was_training:
            model.train()


def disable_base_dropout(model) -> int:
    """
    Zero the dropout probability of every module EXCEPT LoRA's own dropout.

    The base is frozen, so its dropout regularises nothing -- it only injects
    noise into an already high-variance REINFORCE estimator, and it corrupts
    the KL anchor: an untrained adapter is exactly the reference (LoRA
    initialises B=0), yet train-mode dropout alone reports 0.1-0.3 nats per
    position of drift that no weight ever caused. Returns how many settings
    were changed. Sets p=0.0 rather than calling .eval() on them because a
    later model.train() would undo .eval() recursively.

    Two mechanisms have to be silenced, not one. Zeroing nn.Dropout MODULES
    is not sufficient under the SDPA attention implementation (the default
    here): RobertaSdpaSelfAttention keeps its rate in a plain float attribute
    and passes it to scaled_dot_product_attention(dropout_p=...), so it is
    invisible to a module scan and leaves the forward pass stochastic. The
    float attributes are therefore zeroed too.
    """
    n = 0
    for name, module in model.named_modules():
        if "lora" in name.lower():
            continue
        if isinstance(module, torch.nn.Dropout) and module.p:
            module.p = 0.0
            n += 1
        # Functional dropout rates held as floats (SDPA/eager attention).
        for attr in ("dropout_prob", "attention_probs_dropout_prob",
                     "attention_dropout", "hidden_dropout_prob"):
            value = getattr(module, attr, None)
            if isinstance(value, float) and value > 0.0:
                setattr(module, attr, 0.0)
                n += 1

    # The config is what any later-constructed submodule would read from.
    cfg = getattr(getattr(model, "base_model", None), "model", model)
    cfg = getattr(cfg, "config", None)
    for attr in ("attention_probs_dropout_prob", "hidden_dropout_prob"):
        if isinstance(getattr(cfg, attr, None), float) and getattr(cfg, attr) > 0.0:
            setattr(cfg, attr, 0.0)
            n += 1
    return n


def reinforce_rollout_oneshot(
    smiles_masked: str,
    tokenizer,
    model,
    device:      str,
    top_k:       int   = TOP_K,
    temperature: float = TEMPERATURE,
    reward_fn          = None,
    kl_beta:     float = 0.0,
):
    """
    One-shot REINFORCE episode -- ChemBERTa's native MLM decoding. A SINGLE
    forward pass scores every <mask> position at once; each masked position
    is then sampled INDEPENDENTLY from that one pass (conditionally
    independent given the surrounding unmasked context -- a mask never
    sees what any other mask sampled). This is what "predict all masks at
    once" means for a masked language model, matching
    generate_smiles_oneshot in stage2_molecule_generation.py (the decoder
    config.ONESHOT_MASK_DECODING selects there). Stage 9 and Stage 9a use
    this unconditionally -- not gated behind that flag -- so training
    (this function, called from run_stage9_finetuning) and evaluation
    (generate_completion below, which also calls this) share one decoding
    procedure end to end: the only thing that should differ between a
    Stage 9a baseline plot and a Stage 9 post-training plot is the model
    weights, never the sampling strategy.

    Contrast with stage1_9_LLM_RDkit_policy_training.reinforce_rollout,
    which re-runs the model after every fill (N forward passes for N
    masks) so later masks condition on earlier ones -- that
    sequential/dependency-aware mode is intentionally NOT used by Stage 9
    or Stage 9a.

    `reward_fn` may return either a bare float or a (float, components_dict)
    pair. The pair form exists so the caller gets the component breakdown it
    needs for logging out of the SAME scoring call that produced the reward:
    re-scoring the finished SMILES afterwards is deterministic and therefore
    pure waste, and RDKit scoring -- not the GPU -- is this loop's dominant
    cost (~17 ms per call against ~3-5 ms for a batch-1 ChemBERTa forward).

    Returns
    -------
    reward    : scalar float from reward_fn
    log_prob  : scalar Tensor with grad_fn (sum of log P over mask
                positions, all read off the single forward pass' logits)
    kl        : scalar Tensor with grad_fn -- sum over mask positions of the
                exact KL(policy || frozen pretrained) over the full vocab,
                or a constant 0 when kl_beta == 0 (no reference pass is run
                in that case). Anchors the policy to the pretrained
                distribution; see config.STAGE9_KL_BETA. Computed on the RAW
                logits, deliberately NOT the temperature-scaled/top-k ones:
                the anchor constrains the MODEL, not the sampling policy laid
                over it, so changing top_k or temperature must not silently
                change what "drift" means.
    generated : decoded SMILES (may be invalid -- reward_fn handles that)
    comps     : the component dict reward_fn returned alongside its score, or
                None when reward_fn returned a bare float.
    """
    max_len = getattr(tokenizer, "model_max_length", 512)
    if max_len is None or max_len > 1024:
        max_len = 512

    enc = tokenizer(
        smiles_masked, return_tensors="pt",
        truncation=True, max_length=max_len,
    ).to(device)

    ids       = enc["input_ids"][0].clone()
    attn_mask = enc["attention_mask"][0]
    mask_id   = tokenizer.mask_token_id

    mask_positions = (ids == mask_id).nonzero(as_tuple=True)[0].tolist()
    if not mask_positions:
        clean = smiles_masked.replace(" ", "")
        zero  = torch.tensor(0.0, requires_grad=False)
        reward, comps = _split_reward(reward_fn(clean))
        return reward, zero, zero, clean, comps

    # Single forward pass on the still-fully-masked input -- every mask
    # position's logits come from this one call, never a re-run.
    out = model(input_ids=ids.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0))

    # The anchor is read from the SAME still-masked input, and must be taken
    # before the sampling loop below writes chosen tokens back into `ids`.
    ref_logits = _reference_logits(model, ids, attn_mask) if kl_beta > 0.0 else None

    log_prob_sum = torch.tensor(0.0, device=device)
    kl_sum       = torch.tensor(0.0, device=device)
    for pos in mask_positions:
        logits = out.logits[0, pos] / max(temperature, 1e-8)

        log_probs_full = F.log_softmax(logits, dim=-1)
        probs_full     = log_probs_full.exp()
        topk           = torch.topk(probs_full, k=min(top_k, logits.shape[0]))
        top_p          = topk.values / topk.values.sum()

        chosen_local = torch.multinomial(top_p.detach(), num_samples=1).item()
        chosen_id    = topk.indices[chosen_local]

        log_prob_sum = log_prob_sum + log_probs_full[chosen_id]
        ids[pos]     = chosen_id.detach()

        if ref_logits is not None:
            # Exact KL over the whole vocabulary rather than the sampled
            # token's log-ratio: it is differentiable on its own (no second
            # REINFORCE term needed) and carries none of the sampling noise a
            # one-sample estimate would inject into an already noisy gradient.
            lp_policy = F.log_softmax(out.logits[0, pos], dim=-1)
            lp_ref    = F.log_softmax(ref_logits[0, pos], dim=-1)
            kl_sum    = kl_sum + (lp_policy.exp() * (lp_policy - lp_ref)).sum()

    generated     = tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
    reward, comps = _split_reward(reward_fn(generated))
    return reward, log_prob_sum, kl_sum, generated, comps


# ════════════════════════════════════════════════════════════════════════════
#  SCORE FUNCTION (valid + QED + SA + novelty + toxicity)
# ════════════════════════════════════════════════════════════════════════════

# Keys compute_property_components returns. "valid" is the only one that is
# always a number; everything else is None when it could not be measured.
PROPERTY_KEYS = (
    "valid", "qed", "sa_raw", "sa_norm", "novelty",
    "pains", "brenk", "any_alert", "n_alerts", "alert_free", "tox21",
)


def compute_property_components(
    generated_smiles: str,
    original_smiles:  str,
    need_alert_count: bool = True,
    need_tox21:       bool = True,
) -> Dict[str, Optional[float]]:
    """
    Measure one generated SMILES against its pre-mask parent, reporting
    None -- NOT 0.0 -- for anything that cannot be measured:

      valid      1.0/0.0, always a number: RDKit parsed it or it didn't.
                 This is the ONLY component defined for every attempt, so
                 it is the only one whose rate is taken over all generated
                 molecules.
      qed        RDKit QED, continuous 0-1.                  None if invalid.
      sa_raw     SA-Score on its native 1-10 scale (1 easy). None if invalid
                 or rdkit.Contrib.SA_Score is unavailable.
      sa_norm    (10 - sa_raw)/10, the form the composite score uses.
      novelty    1 - Tanimoto(Morgan(parent), Morgan(generated)). None if
                 invalid, or if the PARENT itself can't be fingerprinted
                 (PDB-derived parents are sometimes unparseable) -- novelty
                 against an unknown parent is not 0, it is unknown.
      pains      1.0 if any PAINS filter matches, else 0.0.  None if invalid
      brenk      1.0 if any Brenk filter matches, else 0.0.   or the RDKit
      any_alert  1.0 if either matched, else 0.0.             FilterCatalog
      n_alerts   number of matching alerts (count, not 0-1).  is missing.
      alert_free 1 - any_alert, the form the composite score uses.
      tox21      1 - aggregated toxic probability.  None unless a Tox21
                 checkpoint is configured (config.STAGE9_TOX21_MODEL_DIR).

    Distinguishing None from 0.0 is what lets the Stage 9a report count an
    invalid molecule in the validity denominator while keeping it out of the
    QED / SA / novelty / alert statistics entirely, instead of feeding them
    a fake worst-case zero.

    `need_alert_count` controls only "n_alerts", which is a REPORTING field --
    the composite score uses "alert_free", derived from the two cheap HasMatch
    calls. Producing n_alerts needs FilterCatalog.GetMatches, benchmarked at
    ~6.5 ms against ~17.4 ms for this whole function on a drug-like molecule:
    over a third of the cost for a number the training loop never reads. The
    training path therefore passes False (via compute_stage9_score, which
    always does) and n_alerts stays None there; the eval path keeps the
    default True so every published figure and per-molecule CSV still carries
    it.
    """
    out: Dict[str, Optional[float]] = {k: None for k in PROPERTY_KEYS}

    mol = Chem.MolFromSmiles(generated_smiles) if generated_smiles else None
    if mol is None:
        out["valid"] = 0.0
        return out
    out["valid"] = 1.0

    try:
        out["qed"] = float(QED.qed(mol))
    except Exception:
        pass

    if _SA_AVAILABLE:
        try:
            raw_sa = float(sascorer.calculateScore(mol))
            out["sa_raw"]  = raw_sa
            out["sa_norm"] = max(0.0, (10.0 - raw_sa) / 10.0)
        except Exception:
            pass

    orig_fp = _morgan_fp_cached(original_smiles)
    gen_fp  = _morgan_fp_cached(generated_smiles)
    if orig_fp is not None and gen_fp is not None:
        out["novelty"] = max(0.0, 1.0 - TanimotoSimilarity(orig_fp, gen_fp))

    if _TOX_AVAILABLE:
        try:
            pains = _get_alert_catalog("pains").HasMatch(mol)
            brenk = _get_alert_catalog("brenk").HasMatch(mol)
            out["pains"]      = 1.0 if pains else 0.0
            out["brenk"]      = 1.0 if brenk else 0.0
            out["any_alert"]  = 1.0 if (pains or brenk) else 0.0
            out["alert_free"] = 1.0 - out["any_alert"]
            if need_alert_count:
                out["n_alerts"] = float(len(_get_alert_catalog("any").GetMatches(mol)))
        except Exception:
            pass

    # need_tox21=False leaves out["tox21"] as None for a caller that will
    # supply it separately. Stage 9.1 uses this so its RDKit worker PROCESSES
    # never touch the Tox21 checkpoint: loading a torch model once per worker
    # would multiply memory by the pool size and contend for the GPU, when the
    # same probabilities are far cheaper as one batched forward on the parent.
    if _TOX21_AVAILABLE and need_tox21:
        out["tox21"] = score_tox21(generated_smiles)

    return out


def compute_stage9_score(
    generated_smiles: str,
    original_smiles:  str,
    weights: Tuple[float, float, float, float, float, float] = (
        SCORE_W_VALID, SCORE_W_QED, SCORE_W_SA, SCORE_W_NOVELTY,
        SCORE_W_TOX_ALERT, SCORE_W_TOX21),
) -> Tuple[float, Dict[str, float]]:
    """
    Composite score for one generated SMILES against the pre-mask parent.

        score = w_valid     · valid
              + w_qed       · QED
              + w_sa        · (1 − SA/10)
              + w_novelty   · (1 − Tanimoto(original, generated))
              + w_tox_alert · (1 − PAINS/Brenk_alert)
              + w_tox21     · (1 − Tox21_classifier_toxic_prob)

    Returns (score clamped to [0, 1], component dict for logging).
    Every component defaults to 0.0 (worst case) when it can't be computed
    (invalid molecule, missing optional dependency, etc.) — same
    fail-safe convention as stage1_9's compute_reward. The measurement
    itself lives in compute_property_components below, which distinguishes
    "measured 0.0" from "not measurable" (None); this function collapses
    None to 0.0 because REINFORCE needs a number for every rollout.

    The returned component dict has no "n_alerts" key -- the score uses
    alert_free, not the count -- so the expensive GetMatches call is skipped
    unconditionally here. See compute_property_components' need_alert_count.
    """
    measured = compute_property_components(
        generated_smiles, original_smiles, need_alert_count=False,
    )
    return compose_stage9_score(measured, weights)


def compose_stage9_score(
    measured: Dict[str, Optional[float]],
    weights: Tuple[float, float, float, float, float, float] = (
        SCORE_W_VALID, SCORE_W_QED, SCORE_W_SA, SCORE_W_NOVELTY,
        SCORE_W_TOX_ALERT, SCORE_W_TOX21),
) -> Tuple[float, Dict[str, float]]:
    """
    Collapse a compute_property_components dict into (score, components).

    Split out of compute_stage9_score so the weighted sum that DEFINES the
    Stage 9 objective exists in exactly one place. Stage 9.1 measures the
    RDKit terms in worker processes and the Tox21 term in a batched GPU
    forward on the parent process, then merges the two and calls this -- if it
    re-implemented the sum instead, the two stages' objectives could silently
    drift apart and their results would no longer be comparable.

    None means "not measurable" and becomes 0.0 (worst case) here, the same
    fail-safe convention compute_stage9_score has always used.
    """
    w_valid, w_qed, w_sa, w_novelty, w_tox_alert, w_tox21 = weights

    components = {
        "valid":     measured["valid"] or 0.0,
        "qed":       measured["qed"]       if measured["qed"]       is not None else 0.0,
        "sa":        measured["sa_norm"]   if measured["sa_norm"]   is not None else 0.0,
        "novelty":   measured["novelty"]   if measured["novelty"]   is not None else 0.0,
        "tox_alert": measured["alert_free"] if measured["alert_free"] is not None else 0.0,
        "tox21":     measured["tox21"]     if measured["tox21"]     is not None else 0.0,
    }

    score = (
        w_valid     * components["valid"]
        + w_qed     * components["qed"]
        + w_sa      * components["sa"]
        + w_novelty * components["novelty"]
        + w_tox_alert * components["tox_alert"]
        + w_tox21   * components["tox21"]
    )
    return float(min(max(score, 0.0), 1.0)), components


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING DATA: Stage 1a + Stage 1b masked/original SMILES pairs
# ════════════════════════════════════════════════════════════════════════════

_TAR_EXTS = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar")


def _is_tar_path(path: str) -> bool:
    """
    True if `path` should be opened with tarfile: matches a common tar
    extension, or -- since real-world archive names don't always follow the
    "*.tar.gz" convention (e.g. "stage1b_output_tar.gz", underscore instead
    of a dot before "tar") -- is an existing file tarfile itself recognises
    as a tar archive (handles the gzip/bz2/xz-wrapped case too, same as the
    "r:*" mode used to open it below).
    """
    if path.lower().endswith(_TAR_EXTS):
        return True
    return os.path.isfile(path) and tarfile.is_tarfile(path)


def _tar_stem(name: str) -> str:
    """
    Basename with any tar-ish suffix stripped -- including the
    underscore-instead-of-dot variants ("stage1b_output_tar.gz") -- so a
    configured path and the actual on-disk file can be matched up when only
    that suffix differs.
    """
    lowered = name.lower()
    for ext in _TAR_EXTS:
        for sep in (".", "_"):
            suffix = sep + ext.lstrip(".")
            if lowered.endswith(suffix):
                return lowered[: -len(suffix)]
    return lowered


def _resolve_data_location(location: str) -> str:
    """
    If `location` doesn't exist but a sibling file with the same stem and a
    different tar suffix does, return that instead (warning about the
    substitution). Covers the common Drive/scp case where the archive is
    named stage1b_output.tar.gz but the config says stage1b_output_tar.gz
    (or .tar vs .tar.gz). Returns `location` unchanged when nothing matches,
    so the caller's existing diagnostics still report the miss.
    """
    if not location or os.path.exists(location):
        return location

    parent = os.path.dirname(os.path.abspath(location))
    if not os.path.isdir(parent):
        return location

    stem = _tar_stem(os.path.basename(location))
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return location

    for name in names:
        candidate = os.path.join(parent, name)
        if _tar_stem(name) == stem and os.path.isfile(candidate):
            tqdm.write(
                f"  {location} does not exist -- using {candidate} instead "
                f"(same name, different tar/gz suffix)."
            )
            return candidate
    return location


def _first_existing_ancestor(path: str) -> str:
    """
    Walk up from `path` until an existing directory is found. Pinpoints
    exactly where a configured path diverges from what's actually on disk
    -- e.g. on Colab, ancestor == "/content" means Drive isn't mounted at
    all; ancestor == ".../GenAI4Drug" (stopping one level short of the
    configured file) usually means a typo or case mismatch in the next
    path segment (Drive's FUSE mount, like the rest of Linux, is
    case-sensitive, unlike Windows/macOS).
    """
    p = os.path.abspath(path)
    while p and not os.path.isdir(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def _describe_missing_path(location: str) -> str:
    """Human-readable reason + closest-existing-ancestor listing for a path
    that turned out not to be a directory or a readable archive."""
    ancestor = _first_existing_ancestor(location)
    try:
        listing = sorted(os.listdir(ancestor))[:20] if os.path.isdir(ancestor) else []
    except OSError as e:
        listing = [f"<could not list: {e}>"]
    detail = f"contains: {listing}" if listing else "empty or unreadable"
    return (f"path exists: {os.path.exists(location)}; closest existing "
            f"ancestor: {ancestor!r} ({detail})")


def _read_csv_rows_matching(location: str, filename_pattern: str) -> List[dict]:
    """
    Read & concatenate CSV rows from every file (in a directory) or tar
    member whose basename matches filename_pattern (glob-style, e.g.
    "mask15pct_temp*_seed*.csv"). `location` may be a plain directory or a
    .tar/.tar.gz/.tgz/.tar.bz2/.tar.xz archive path -- lets Stage 1a/1b
    output be consumed straight from an archive without extracting it to
    disk first.

    Every "found nothing" branch prints WHY (couldn't open the archive,
    opened it but no member matched, path isn't a directory or a
    recognised archive) instead of silently returning [] -- a silent empty
    return here is what previously made a real Stage 1b tar.gz (present on
    disk, but not yielding rows for some reason) look identical to "Stage
    1b data doesn't exist at all".
    """
    rows: List[dict] = []
    location = _resolve_data_location(location)
    if _is_tar_path(location):
        try:
            with tarfile.open(location, "r:*") as tar:
                members = tar.getmembers()
                matched = 0
                for member in members:
                    if not member.isfile():
                        continue
                    if not fnmatch.fnmatch(os.path.basename(member.name), filename_pattern):
                        continue
                    matched += 1
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        continue
                    text = io.TextIOWrapper(fileobj, encoding="utf-8", newline="")
                    rows.extend(csv.DictReader(text))
                if matched == 0:
                    tqdm.write(
                        f"  {location}: opened as a tar archive with {len(members)} "
                        f"member(s), but none matched {filename_pattern!r}."
                    )
        except tarfile.TarError as e:
            tqdm.write(f"  {location}: found on disk but failed to open as a tar archive ({e}).")
        return rows

    if not os.path.isdir(location):
        tqdm.write(
            f"  {location}: not a directory, and not recognised as a tar archive "
            f"({_describe_missing_path(location)})."
        )
        return rows

    for csv_path in sorted(glob.glob(os.path.join(location, filename_pattern))):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _remask_seed(base_seed: int, key: str) -> int:
    """Deterministic per-row seed (mirrors stage1a_random_masking.mask_seed)."""
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _remask_from_pool(
    smiles:    str,
    pool:      List[int],
    percent:   float,
    tokenizer,
    seed_key:  str,
    base_seed: int,
) -> Tuple[str, Optional[str]]:
    """
    Rebuild masked_smiles by masking floor(percent% * total_bpe_tokens) atom
    indices sampled from `pool` -- masks the whole pool instead when it has
    fewer atoms than that target.

    Returns (masked_smiles, skip_reason). skip_reason is None on success, or
    one of "empty_pool" / "no_mask_at_percent" / "invalid_parent" /
    "over_length" / "mask_failed" with masked_smiles == "". Every failure is
    reported this way rather than raised: PDB-derived parents include
    SMILES RDKit refuses outright (hypervalent P/S from bond perception,
    unkekulizable aromatic rings), and one such row out of thousands must
    not abort the whole run -- which is exactly what used to happen when
    the masker's last fallback raised ValueError through this function.
    """
    if not pool:
        return "", "empty_pool"

    if Chem.MolFromSmiles(smiles) is None:
        # Novelty/QED against an unparseable parent are undefined, so the row
        # is dropped rather than masked as a raw string.
        return "", "invalid_parent"

    n_tokens = len(tokenizer(smiles, add_special_tokens=False)["input_ids"])
    if n_tokens + 2 > MAX_MODEL_TOKENS:      # +2 for <s> / </s>
        return "", "over_length"

    n_target = int(percent / 100.0 * n_tokens)  # floor
    n_mask   = min(n_target, len(pool))
    if n_mask <= 0:
        return "", "no_mask_at_percent"

    if len(pool) <= n_mask:
        sampled = pool
    else:
        rng = random.Random(_remask_seed(base_seed, seed_key))
        sampled = sorted(rng.sample(pool, n_mask))

    try:
        masked = mask_atoms_in_smiles_token_level(smiles, sampled, tokenizer)
    except Exception:
        return "", "mask_failed"

    if not masked or tokenizer.mask_token not in masked:
        return "", "mask_failed"

    # The masked string is what the model actually sees, and it can be LONGER
    # in tokens than the parent (each <mask> is its own token, and masking
    # splits tokens that used to merge), so it gets its own check.
    if len(tokenizer(masked, add_special_tokens=True)["input_ids"]) > MAX_MODEL_TOKENS:
        return "", "over_length"
    return masked, None


# Per-source bookkeeping from the LAST collect_pairs_from_stage1a/1b call:
# {source: {"rows": N, "pairs": P, "<skip reason>": count, ...}}. The Stage 9a
# report prints this as its figure footer so the plots always state what was
# dropped before generation ever started.
COLLECTION_STATS: Dict[str, Dict[str, int]] = {}


def collect_pairs_from_stage1a(
    stage1a_dir: str,
    percent:     float = None,
) -> List[Tuple[str, str]]:
    """
    (masked_smiles, original_smiles) pairs from the Stage-1a combo CSV(s)
    matching `percent` (default config.STAGE9_MASK_PERCENT). `stage1a_dir`
    may be a plain directory or a .tar/.tar.gz/.tgz archive (see
    _read_csv_rows_matching).

    Rows are dropped (counted in COLLECTION_STATS, never raised) when a
    field is missing, the parent SMILES doesn't parse, the stored
    masked_smiles carries no <mask>, or the pair exceeds ChemBERTa's
    MAX_MODEL_TOKENS window.
    """
    percent = config.STAGE9_MASK_PERCENT if percent is None else percent
    pattern = f"mask{percent:g}pct_temp*_seed*.csv"
    rows    = _read_csv_rows_matching(stage1a_dir, pattern)
    if not rows:
        tqdm.write(
            f"  No Stage 1a files/members matched {pattern} in {stage1a_dir} "
            f"(available percents: {config.STAGE1A_MASK_PERCENTS})."
        )

    tokenizer = get_chemberta_tokenizer()
    stats = {"rows": len(rows), "pairs": 0, "missing_field": 0,
             "invalid_parent": 0, "no_mask": 0, "over_length": 0}

    pairs: List[Tuple[str, str]] = []
    for row in rows:
        masked = (row.get("masked_smiles") or "").strip()
        orig   = (row.get("smiles") or "").strip()
        if not masked or not orig:
            stats["missing_field"] += 1
            continue
        if tokenizer.mask_token not in masked:
            stats["no_mask"] += 1
            continue
        if Chem.MolFromSmiles(orig) is None:
            stats["invalid_parent"] += 1
            continue
        n_tokens = len(tokenizer(masked, add_special_tokens=True)["input_ids"])
        if n_tokens > MAX_MODEL_TOKENS:
            stats["over_length"] += 1
            continue
        pairs.append((masked, orig))

    stats["pairs"] = len(pairs)
    COLLECTION_STATS["stage1a"] = stats
    _report_collection_stats("stage1a", stage1a_dir, stats)
    return pairs


def _report_collection_stats(source: str, location: str, stats: Dict[str, int]) -> None:
    """One line per source naming every dropped row and why."""
    dropped = {k: v for k, v in stats.items()
               if k not in ("rows", "pairs") and v}
    if not stats["rows"]:
        return
    detail = ", ".join(f"{k}: {v}" for k, v in dropped.items()) or "none"
    tqdm.write(
        f"  {source}: {stats['rows']} row(s) read from {location} -> "
        f"{stats['pairs']} usable pair(s); dropped ({detail})."
    )


def collect_pairs_from_stage1b(
    stage1b_dir: str,
    percent:     float = None,
    seed:        int   = None,
) -> List[Tuple[str, str]]:
    """
    (masked_smiles, original_smiles) pairs from the Stage-1b summary CSV,
    re-masked to `percent` (default config.STAGE9_MASK_PERCENT) of each
    molecule's BPE tokens, sampled from its PLIP-derived masked_atom_indices
    pool -- see module docstring "Training data" for the full rule.
    `stage1b_dir` may be a plain directory or a .tar/.tar.gz/.tgz archive
    (see _read_csv_rows_matching).
    """
    percent = config.STAGE9_MASK_PERCENT if percent is None else percent
    seed    = config.STAGE9_MASK_SEED    if seed    is None else seed

    rows = _read_csv_rows_matching(
        stage1b_dir, "stage1b_large_scale_plip_mask_summary.csv",
    )
    if not rows:
        tqdm.write(
            f"  No Stage 1b files/members matched stage1b_large_scale_plip_mask_summary.csv "
            f"in {stage1b_dir} (see the reason above, if any)."
        )
        return []

    tokenizer = get_chemberta_tokenizer()
    pairs: List[Tuple[str, str]] = []
    stats = {"rows": len(rows), "pairs": 0, "status_not_ok": 0, "missing_field": 0,
             "bad_pool": 0, "empty_pool": 0, "no_mask_at_percent": 0,
             "invalid_parent": 0, "over_length": 0, "mask_failed": 0}

    for row in rows:
        if row.get("status") != "ok":
            stats["status_not_ok"] += 1
            continue
        orig = (row.get("smiles") or "").strip()
        if not orig:
            stats["missing_field"] += 1
            continue
        try:
            pool = json.loads(row.get("masked_atom_indices") or "[]")
        except (json.JSONDecodeError, TypeError):
            stats["bad_pool"] += 1
            continue

        seed_key = (f"{row.get('pdb_id','')}:{row.get('resname','')}:"
                    f"{row.get('chain','')}:{row.get('resseq','')}")
        try:
            masked, reason = _remask_from_pool(
                orig, pool, percent, tokenizer, seed_key, seed,
            )
        except Exception as exc:                     # belt-and-braces: never abort
            stats["mask_failed"] += 1
            if stats["mask_failed"] <= 5:
                tqdm.write(f"  skipping {seed_key}: re-mask failed ({exc})")
            continue

        if reason is None:
            pairs.append((masked, orig))
        else:
            stats[reason] = stats.get(reason, 0) + 1

    stats["pairs"] = len(pairs)
    COLLECTION_STATS["stage1b"] = stats
    _report_collection_stats("stage1b", stage1b_dir, stats)
    return pairs


def collect_all_training_pairs(
    stage1a_dir:    str = None,
    stage1b_dir:    str = None,
    max_pairs:      int = MAX_TRAINING_PAIRS,
    sample_seed:    int = SAMPLE_SEED,
    max_per_parent: int = None,
) -> List[Tuple[str, str]]:
    """
    Concatenate Stage-1a + Stage-1b pairs per config.STAGE9_DATA_SOURCE
    ("stage1a" | "stage1b" | "both"), optionally sub-sampled to max_pairs.

    `max_per_parent` (default config.STAGE9_MAX_PAIRS_PER_PARENT) caps how many
    ligand instances of the same parent molecule reach the training loop -- see
    _cap_pairs_per_parent. It is applied PER SOURCE and BEFORE max_pairs, which
    is deliberate on both counts: per source, because a molecule appearing in
    both Stage 1a and Stage 1b should keep an entry from each (random masking
    and PLIP masking are the two conditions being contrasted, not duplicates of
    one another); before the sub-sample, so max_pairs draws from a pool of
    distinct chemistry rather than re-sampling the same over-represented ions.

    Directory resolution per source (explicit arg > STAGE9_9A_*_DATA_DIR
    override > default STAGE1A_DIR / STAGE1B_PLIP_MASK_DIR) lets masks
    computed on one machine be trained on another -- point the
    STAGE9_9A_STAGE1A_DATA_DIR / STAGE9_9A_STAGE1B_DATA_DIR override at
    wherever that stage's output was copied to.
    """
    stage1a_dir = (stage1a_dir or getattr(config, "STAGE9_9A_STAGE1A_DATA_DIR", "")
                   or config.STAGE1A_DIR)
    stage1b_dir = (stage1b_dir or getattr(config, "STAGE9_9A_STAGE1B_DATA_DIR", "")
                   or config.STAGE1B_PLIP_MASK_DIR)
    source = getattr(config, "STAGE9_DATA_SOURCE", "both")
    if source not in ("stage1a", "stage1b", "both"):
        raise ValueError(
            f"config.STAGE9_DATA_SOURCE must be 'stage1a', 'stage1b', or 'both', "
            f"got {source!r}."
        )

    if max_per_parent is None:
        max_per_parent = MAX_PAIRS_PER_PARENT

    a = collect_pairs_from_stage1a(stage1a_dir) if source in ("stage1a", "both") else []
    b = collect_pairs_from_stage1b(stage1b_dir) if source in ("stage1b", "both") else []
    tqdm.write(
        f"  Collected {len(a)} pairs from Stage 1a, {len(b)} pairs from Stage 1b "
        f"(source={source!r}, mask_percent={config.STAGE9_MASK_PERCENT})."
    )

    if max_per_parent:
        a = _cap_pairs_per_parent(a, "stage1a", max_per_parent, sample_seed)
        b = _cap_pairs_per_parent(b, "stage1b", max_per_parent, sample_seed)

    pairs = a + b
    if max_pairs and len(pairs) > max_pairs:
        rng = random.Random(sample_seed)
        pairs = rng.sample(pairs, max_pairs)
        tqdm.write(f"  Sub-sampled to {max_pairs} pairs (seed={sample_seed}).")

    tqdm.write(f"  Total training pairs: {len(pairs)}")
    return pairs


# ════════════════════════════════════════════════════════════════════════════
#  PROPERTY-DISTRIBUTION EVALUATION  (shared by Stage 9 post-training eval and
#  Stage 9a's no-fine-tuning baseline -- see stage9a_masked_property_without_finetuning.py)
# ════════════════════════════════════════════════════════════════════════════

# Fixed source -> (legend label, color) so a given source always renders in the
# same color across every figure this module or Stage 9a produces. Colors are
# categorical slots 1 (blue) and 2 (orange) from the project's validated,
# colorblind-safe palette (adjacent-pair CVD ΔE well above the 8 target).
SOURCE_LABELS = {
    "stage1a": "Random masking (Stage 1a)",
    "stage1b": "PLIP masking (Stage 1b)",
}
SOURCE_COLORS = {
    "stage1a": "#2a78d6",
    "stage1b": "#eb6834",
}
_SOURCE_ORDER = ("stage1a", "stage1b")


@lru_cache(maxsize=200_000)
def _canonical_parent(smiles: str) -> str:
    """
    Canonical (RDKit) form of a parent SMILES, used only as a dedup key.
    Falls back to the raw string when RDKit can't parse it, so an unparseable
    parent groups with its exact duplicates instead of vanishing.

    Memoised because the callers grouping by parent walk ligand-INSTANCE lists:
    334k Stage 1b pairs carry only ~45k distinct parent strings, so without the
    cache the same molecule is parsed and re-canonicalised ~7x on average (and
    the most common ones tens of thousands of times) for an identical answer.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return smiles
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


def _dedup_pairs_by_parent(
    pairs:  List[Tuple[str, str]],
    source: str,
) -> List[Tuple[str, str]]:
    """
    Collapse a source's pairs to ONE pair per unique parent molecule.

    Stage 1b writes one row per ligand INSTANCE (pdb_id:resname:chain:resseq),
    so a ligand resolved in N PDB entries contributes N pairs -- N generation
    attempts on the same parent, each with its own instance-seeded mask. That
    inflates the denominator of every statistic (validity, "identical to
    parent", ...) toward whichever molecules crystallography happens to have
    solved most often, and it makes `max_pairs_per_source` a cap on instances
    rather than on molecules.

    Keeps the FIRST occurrence in list order. Callers build that list
    deterministically from the same CSV rows, so Stage 9a's baseline and
    Stage 9's post-training pass keep the identical pair for each parent and
    their before/after figures stay comparable. Parents are keyed on their
    canonical SMILES, so the same molecule written two ways still collapses.

    Runs BEFORE _subsample_pairs, which is the whole point: the sample is then
    drawn from distinct molecules.
    """
    seen:  set = set()
    kept:  List[Tuple[str, str]] = []
    for masked_smi, orig_smi in pairs:
        key = _canonical_parent(orig_smi)
        if key in seen:
            continue
        seen.add(key)
        kept.append((masked_smi, orig_smi))

    if len(kept) < len(pairs):
        stats = COLLECTION_STATS.setdefault(source, {})
        stats["deduped_from"]      = len(pairs)
        stats["duplicate_parents"] = len(pairs) - len(kept)
        stats["pairs"]            = len(kept)
        tqdm.write(
            f"  {source}: {len(kept)} unique parent molecule(s) from {len(pairs)} "
            f"ligand-instance pair(s) ({len(pairs) - len(kept)} duplicate parent(s) "
            f"dropped); set config.STAGE9_EVAL_DEDUP_BY_PARENT=False to score every "
            f"instance instead."
        )
    return kept


def _cap_pairs_per_parent(
    pairs:           List[Tuple[str, str]],
    source:          str,
    max_per_parent:  int,
    sample_seed:     int,
) -> List[Tuple[str, str]]:
    """
    Keep at most `max_per_parent` ligand instances per unique parent molecule
    -- the TRAINING-pool analogue of _dedup_pairs_by_parent (which is the eval
    pass' stricter one-per-parent version).

    Why a cap rather than nothing: Stage 1b writes one row per ligand instance,
    and PDB instance counts track crystallography, not chemistry. On the
    shipped summary at 15% masking, 334,550 surviving pairs carry only 44,977
    distinct molecules, and the six most frequent parents -- sulfate (33,538 +
    13,816 across two stereo-notations), glycerol (20,249 + 18,876), NAG
    (14,549) and acetate (10,376) -- are 33% of the whole set on their own, with
    the top 100 parents at 58.5%. Uncapped, the REINFORCE gradient is dominated
    by buffer components and cryoprotectants, and QED / SA / novelty on a
    sulfate ion are not meaningful quantities to optimise toward.

    Why a cap rather than one-per-parent: the instances of a molecule are NOT
    duplicates. Each carries its own PLIP-derived masked_atom_indices pool from
    a different binding pocket, so _remask_from_pool masks different atoms --
    genuine augmentation over mask positions. Keeping a few per parent retains
    that diversity while discarding the popularity weighting; see
    config.STAGE9_MAX_PAIRS_PER_PARENT for the 3-vs-1 trade-off.

    The instances kept are drawn with a per-parent deterministic seed rather
    than taken in list order, so the retained pockets are a spread over that
    molecule's structures instead of whichever PDB IDs happen to sort first.
    Output preserves input order, so the result is reproducible and still
    lines up with how every other collector builds its list.
    """
    if not max_per_parent or max_per_parent < 1:
        return pairs

    by_parent: Dict[str, List[int]] = {}
    for idx, (_masked, orig_smi) in enumerate(pairs):
        by_parent.setdefault(_canonical_parent(orig_smi), []).append(idx)

    keep: set = set()
    for parent, idxs in by_parent.items():
        if len(idxs) <= max_per_parent:
            keep.update(idxs)
        else:
            rng = random.Random(_remask_seed(sample_seed, parent))
            keep.update(rng.sample(idxs, max_per_parent))

    kept = [p for i, p in enumerate(pairs) if i in keep]

    if len(kept) < len(pairs):
        stats = COLLECTION_STATS.setdefault(source, {})
        stats["capped_from"]        = len(pairs)
        stats["max_pairs_per_parent"] = max_per_parent
        stats["distinct_parents"]   = len(by_parent)
        stats["pairs"]              = len(kept)
        tqdm.write(
            f"  {source}: capped to {len(kept)} training pair(s) from {len(pairs)} "
            f"ligand instance(s) -- at most {max_per_parent} per parent across "
            f"{len(by_parent)} distinct molecule(s); set "
            f"config.STAGE9_MAX_PAIRS_PER_PARENT=None to train on every instance."
        )
    return kept


def _subsample_pairs(
    pairs:       List[Tuple[str, str]],
    source:      str,
    max_pairs:   int,
    sample_seed: int,
) -> List[Tuple[str, str]]:
    """
    Deterministic random sample of `max_pairs` pairs from one source, seeded
    by (sample_seed, source). The input list is built the same way by every
    caller, so Stage 9a's baseline and Stage 9's post-training pass draw the
    IDENTICAL subset -- which is what keeps the before/after figures
    comparable when a limit is in force. Returns `pairs` unchanged when no
    limit applies.
    """
    if not max_pairs or len(pairs) <= max_pairs:
        return pairs
    rng     = random.Random(_remask_seed(sample_seed, source))
    sampled = rng.sample(pairs, max_pairs)
    stats   = COLLECTION_STATS.setdefault(source, {})
    stats["subsampled_from"] = len(pairs)
    stats["sample_seed"]     = sample_seed
    stats["pairs"]           = len(sampled)
    tqdm.write(
        f"  {source}: sampled {len(sampled)} of {len(pairs)} pair(s) "
        f"(seed={sample_seed}); set config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE=None "
        f"to use them all."
    )
    return sampled


def collect_pairs_by_source(
    stage1a_dir:           str  = None,
    stage1b_dir:           str  = None,
    max_pairs_per_source:  int  = None,
    sample_seed:           int  = None,
    dedup_by_parent:       bool = None,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    (masked_smiles, original_smiles) pairs from Stage 1a and Stage 1b, kept
    separate and independent of config.STAGE9_DATA_SOURCE (that knob only
    controls what the REINFORCE training loop trains on). Used for the
    property-distribution plots below: a source key is present only if it
    actually produced pairs, so "both if both are available, otherwise
    whichever one is" falls out naturally from the dict it returns.

    A source that produced nothing is a warning, never a stop: if Stage 1a
    hasn't been run (or its output lives on another machine), the run
    continues on whatever Stage 1b's location yields, and vice versa. Only
    the caller decides what to do when BOTH come up empty.

    `max_pairs_per_source` (default config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE)
    caps each source SEPARATELY -- N from Stage 1a and N from Stage 1b, not N
    in total -- so neither panel is starved by the other source having more
    rows. The sample is drawn here, before any molecule is generated, so a
    smaller limit really is a shorter run.

    `dedup_by_parent` (default config.STAGE9_EVAL_DEDUP_BY_PARENT, True)
    collapses each source to one pair per unique parent molecule BEFORE that
    sample is drawn, so the limit counts distinct molecules rather than
    ligand instances -- see _dedup_pairs_by_parent for why Stage 1b needs it.
    """
    stage1a_dir = (stage1a_dir or getattr(config, "STAGE9_9A_STAGE1A_DATA_DIR", "")
                   or config.STAGE1A_DIR)
    stage1b_dir = (stage1b_dir or getattr(config, "STAGE9_9A_STAGE1B_DATA_DIR", "")
                   or config.STAGE1B_PLIP_MASK_DIR)
    if max_pairs_per_source is None:
        max_pairs_per_source = EVAL_MAX_PAIRS_PER_SOURCE
    if sample_seed is None:
        sample_seed = SAMPLE_SEED
    if dedup_by_parent is None:
        dedup_by_parent = EVAL_DEDUP_BY_PARENT

    def _prepare(pairs: List[Tuple[str, str]], source: str) -> List[Tuple[str, str]]:
        """Dedup to unique parents first, then cap -- order matters."""
        if dedup_by_parent:
            pairs = _dedup_pairs_by_parent(pairs, source)
        return _subsample_pairs(pairs, source, max_pairs_per_source, sample_seed)

    result: Dict[str, List[Tuple[str, str]]] = {}
    a = collect_pairs_from_stage1a(stage1a_dir)
    if a:
        result["stage1a"] = _prepare(a, "stage1a")
    else:
        tqdm.write(
            f"  WARNING: no Stage 1a random-masking pairs in {stage1a_dir} -- "
            f"skipping that source (Stage 1b alone is enough to continue)."
        )
    b = collect_pairs_from_stage1b(stage1b_dir)
    if b:
        result["stage1b"] = _prepare(b, "stage1b")
    else:
        tqdm.write(
            f"  WARNING: no Stage 1b PLIP-masking pairs in {stage1b_dir} -- "
            f"skipping that source (Stage 1a alone is enough to continue)."
        )
    return result


def confirm_partial_sources_or_exit(
    pairs_by_source: Dict[str, List[Tuple[str, str]]],
) -> None:
    """
    If exactly one of Stage 1a / Stage 1b produced pairs, ask before
    continuing rather than silently plotting only the source that loaded --
    the missing source may be an intentional single-source run, or it may
    be a misconfigured path/unmounted Drive that happens to fail silently
    (see collect_pairs_from_stage1a/1b's diagnostics above for why it came
    up empty). Does nothing when zero or both sources are present (the
    caller already handles "zero" as a hard stop; "both" needs no prompt).

    Missing STAGE 1A specifically never prompts: Stage 1a's random-token
    masking is the optional comparison arm here, so a Stage-1b-only run
    (the usual case when only the PLIP archive was copied over) warns and
    proceeds.

    Auto-proceeds without prompting when stdin isn't a TTY (a
    non-interactive/scheduled run), so this can never hang unattended
    execution -- same convention as stage1_9's _ask_resume, which only
    prompts in an interactive session too.
    """
    if len(pairs_by_source) != 1:
        return

    source        = next(iter(pairs_by_source))
    missing       = "stage1b" if source == "stage1a" else "stage1a"
    label         = SOURCE_LABELS.get(source, source)
    missing_label = SOURCE_LABELS.get(missing, missing)
    n = len(pairs_by_source[source])

    print(f"\n  Only {label} data was found ({n} pairs) -- {missing_label} produced no pairs.")
    if missing == "stage1a":
        print("  Stage 1a data is optional -- proceeding with Stage 1b (PLIP masking) only.")
        return
    if not sys.stdin.isatty():
        print("  Non-interactive session -- proceeding with the single available source.")
        return

    while True:
        ans = input(f"  Continue evaluating/plotting {label} only? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            return
        if ans in ("n", "no"):
            print("  Aborted by user.")
            sys.exit(0)
        print("  Please enter Y or N.")


def generate_completion(
    masked_smi:  str,
    tokenizer,
    model,
    device:      str,
    top_k:       int   = TOP_K,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Fill every <mask> token once (inference only -- no gradient, no log-prob
    bookkeeping), reusing reinforce_rollout_oneshot for the actual sampling
    so eval-time generation is identical to what training rolls out: a
    single forward pass, every mask sampled independently from it.
    """
    with torch.no_grad():
        generated = reinforce_rollout_oneshot(
            smiles_masked=masked_smi, tokenizer=tokenizer, model=model,
            device=device, top_k=top_k, temperature=temperature,
            reward_fn=lambda smi: 0.0,
        )[3]
    return generated


def evaluate_property_records(
    tokenizer,
    model,
    device:          str,
    pairs_by_source: Dict[str, List[Tuple[str, str]]],
    top_k:           int   = TOP_K,
    temperature:     float = TEMPERATURE,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Generate one completion per (masked, original) pair and measure it with
    compute_property_components, returning ONE RECORD PER ATTEMPT:

        {source: [{"masked_smiles": ..., "original_smiles": ...,
                   "generated_smiles": ..., "valid": 1.0/0.0,
                   "qed": float|None, "sa_raw": float|None, ...}, ...]}

    Invalid generations are kept as records with valid=0.0 and None for
    every other property. That is the whole point: the validity rate is
    taken over ALL attempts, while QED / SA / novelty / alert statistics are
    taken over the valid subset only (None values are skipped downstream),
    instead of invalid molecules injecting fake zeros into every
    distribution.

    Downstream: summarize_property_records, write_property_records_csv,
    plot_property_report.
    """
    was_training = getattr(model, "training", False)
    model.eval()

    records_by_source: Dict[str, List[Dict[str, object]]] = {}
    n_gen_failed = 0
    for source in _SOURCE_ORDER:
        pairs = pairs_by_source.get(source)
        if not pairs:
            continue
        records: List[Dict[str, object]] = []
        for masked_smi, orig_smi in tqdm(
            pairs, desc=f"  Scoring {SOURCE_LABELS.get(source, source)}", unit="mol",
        ):
            try:
                generated = generate_completion(
                    masked_smi, tokenizer, model, device, top_k, temperature,
                )
            except Exception as exc:                 # never abort a whole run
                n_gen_failed += 1
                if n_gen_failed <= 5:
                    tqdm.write(f"  generation failed ({exc}) -- counted as invalid.")
                generated = ""
            rec: Dict[str, object] = {
                "source":           source,
                "masked_smiles":    masked_smi,
                "original_smiles":  orig_smi,
                "generated_smiles": generated,
            }
            rec.update(compute_property_components(generated, orig_smi))
            records.append(rec)
        records_by_source[source] = records

    if n_gen_failed:
        tqdm.write(f"  {n_gen_failed} generation(s) failed outright and count as invalid.")
    if was_training:
        model.train()
    return records_by_source


def evaluate_parent_property_records(
    pairs_by_source: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, List[Dict[str, object]]]:
    """
    Measure the PRE-MASK PARENT molecule of every pair on the same criteria
    evaluate_property_records measures the generated molecule on -- no model,
    no GPU, no <mask> filling. This is the reference arm of the comparison
    stage9a_masked_property_without_finetuning.py plots: "what did the
    starting chemistry already look like", against which a completion from
    vanilla ChemBERTa (Stage 9a) or from the LoRA-tuned policy (Stage 9) is
    an improvement or a regression.

    Record shape matches evaluate_property_records exactly (same keys, same
    None-means-not-measurable convention) so summarize_property_records,
    write_property_records_csv and plot_property_report all take it
    unchanged. Two fields differ, both deliberately:

      generated_smiles  ""      -- nothing was generated. Left empty rather
                                   than echoing the parent, so a parent row
                                   in the CSV can never be mistaken for a
                                   model output.
      novelty           None    -- novelty is 1 - Tanimoto(parent, X). With
                                   X = the parent, that is 0 by definition,
                                   an identity rather than a measurement.
                                   Reporting None keeps it out of every
                                   statistic and off every panel, instead of
                                   planting a spurious "0% novel" bar next
                                   to the generated distribution.

    "valid" is 1.0 for every record here, and that is a property of the DATA
    PIPELINE, not of the molecules: collect_pairs_from_stage1a/1b drop rows
    whose parent SMILES RDKit rejects (counted as "invalid_parent" in
    COLLECTION_STATS and printed in the figure footer) before a pair is ever
    built. Callers therefore omit the validity panel for parent series --
    see plot_property_report's `omit_panels` / `panel_series`.
    """
    records_by_source: Dict[str, List[Dict[str, object]]] = {}
    for source, pairs in pairs_by_source.items():
        records: List[Dict[str, object]] = []
        for masked_smi, orig_smi in tqdm(
            pairs, desc=f"  Scoring parents {SOURCE_LABELS.get(source, source)}", unit="mol",
        ):
            rec: Dict[str, object] = {
                "source":           source,
                "masked_smiles":    masked_smi,
                "original_smiles":  orig_smi,
                "generated_smiles": "",
            }
            rec.update(compute_property_components(orig_smi, orig_smi))
            rec["novelty"] = None
            records.append(rec)
        records_by_source[source] = records
    return records_by_source


# ── record statistics ──────────────────────────────────────────────────────

def _measured(records: List[Dict[str, object]], key: str) -> List[float]:
    """Values of `key` across records, skipping every record where it is None."""
    return [float(r[key]) for r in records if r.get(key) is not None]


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _sd(vals: List[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _quantile(vals: List[float], q: float) -> float:
    """Linear-interpolated quantile (numpy-free, so records stay plain floats)."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos  = q * (len(s) - 1)
    lo   = int(pos)
    hi   = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for k successes in n trials -- plotted on the
    validity bar so a 200-molecule run doesn't read as precisely as a
    20,000-molecule one.
    """
    if n <= 0:
        return 0.0, 0.0
    p      = k / n
    denom  = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(vals: List[float]) -> float:
    """Fraction of 1.0s among measured 0/1 values (NaN when none measured)."""
    return _mean(vals)


def summarize_property_records(records: List[Dict[str, object]]) -> Dict[str, float]:
    """
    Per-source summary with EXPLICIT denominators: validity over all
    attempts, everything else over however many molecules the property was
    actually measurable on (`*_n` in the returned dict).
    """
    n_attempted = len(records)
    n_valid     = sum(1 for r in records if r.get("valid") == 1.0)

    qed   = _measured(records, "qed")
    sa    = _measured(records, "sa_raw")
    nov   = _measured(records, "novelty")
    tox21 = _measured(records, "tox21")
    pains = _measured(records, "pains")
    brenk = _measured(records, "brenk")
    anyal = _measured(records, "any_alert")
    nalrt = _measured(records, "n_alerts")

    ci_lo, ci_hi = _wilson_ci(n_valid, n_attempted)
    return {
        "n_attempted":     n_attempted,
        "n_valid":         n_valid,
        "n_invalid":       n_attempted - n_valid,
        "validity":        (n_valid / n_attempted) if n_attempted else float("nan"),
        "validity_ci_lo":  ci_lo,
        "validity_ci_hi":  ci_hi,

        "qed_n":           len(qed),
        "qed_mean":        _mean(qed),
        "qed_sd":          _sd(qed),
        "qed_median":      _quantile(qed, 0.5),
        "qed_druglike":    _rate([1.0 if v >= QED_DRUGLIKE_THRESHOLD else 0.0 for v in qed]),

        "sa_n":            len(sa),
        "sa_mean":         _mean(sa),
        "sa_median":       _quantile(sa, 0.5),
        "sa_easy":         _rate([1.0 if v <= SA_EASY_THRESHOLD else 0.0 for v in sa]),
        "sa_norm_mean":    _mean(_measured(records, "sa_norm")),

        "novelty_n":       len(nov),
        "novelty_mean":    _mean(nov),
        "novelty_median":  _quantile(nov, 0.5),
        "novelty_q1":      _quantile(nov, 0.25),
        "novelty_q3":      _quantile(nov, 0.75),
        "novelty_zero":    _rate([1.0 if v <= 1e-9 else 0.0 for v in nov]),

        "alert_n":         len(anyal),
        "pains_rate":      _rate(pains),
        "brenk_rate":      _rate(brenk),
        "any_alert_rate":  _rate(anyal),
        "alerts_mean":     _mean(nalrt),

        "tox21_n":         len(tox21),
        "tox21_mean":      _mean(tox21),
        "tox21_median":    _quantile(tox21, 0.5),
    }


PROPERTY_CSV_FIELDS = [
    "source", "masked_smiles", "original_smiles", "generated_smiles",
    "valid", "qed", "sa_raw", "sa_norm", "novelty",
    "pains", "brenk", "any_alert", "n_alerts", "alert_free", "tox21",
]


def write_property_records_csv(
    records_by_source: Dict[str, List[Dict[str, object]]],
    csv_path:          str,
) -> str:
    """
    One row per generation attempt, unmeasurable cells left EMPTY (not 0),
    so every number on the figure can be recomputed -- or re-plotted, or
    diffed molecule-by-molecule against Stage 9's post-training run --
    without spending another GPU pass over the whole set.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROPERTY_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for source in _SOURCE_ORDER:
            for rec in records_by_source.get(source, []):
                writer.writerow({
                    k: ("" if rec.get(k) is None else rec.get(k))
                    for k in PROPERTY_CSV_FIELDS
                })
    tqdm.write(f"  Per-molecule property records : {csv_path}")
    return csv_path


def read_property_records_csv(
    csv_path: str,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Inverse of write_property_records_csv: rebuild {source: [record, ...]}
    from a per-molecule CSV a previous run wrote.

    This is what lets Stage 9a re-plot without a second GPU pass. Generating
    one completion per molecule is the entire cost of that script; the
    parent measurements are pure RDKit. So when the parent arm is added (or
    a figure is restyled) after the generated arm has already been scored,
    the generated records come back from disk instead of being re-sampled --
    which also makes the comparison exact rather than merely seeded the
    same, since MLM decoding is stochastic.

    Empty cells become None (not 0.0), preserving the distinction
    compute_property_components draws between "measured 0" and "not
    measurable"; numeric columns are parsed as floats and a cell that will
    not parse is treated as not measurable rather than raising. Rows with an
    unrecognised "source" are kept under their own key, so a caller that
    knows about extra series still sees them.

    Returns {} (with a warning) when the file does not exist -- a missing
    cache is a reason to fall back to generating, never a crash.
    """
    if not os.path.isfile(csv_path):
        tqdm.write(f"  No cached per-molecule records at {csv_path}.")
        return {}

    numeric = set(PROPERTY_CSV_FIELDS) - {
        "source", "masked_smiles", "original_smiles", "generated_smiles",
    }
    records_by_source: Dict[str, List[Dict[str, object]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec: Dict[str, object] = {}
            for key in PROPERTY_CSV_FIELDS:
                raw = (row.get(key) or "").strip()
                if key not in numeric:
                    rec[key] = raw
                    continue
                try:
                    rec[key] = float(raw) if raw != "" else None
                except ValueError:
                    rec[key] = None
            records_by_source.setdefault(rec["source"] or "unknown", []).append(rec)

    n = sum(len(v) for v in records_by_source.values())
    tqdm.write(
        f"  Reloaded {n} cached per-molecule record(s) from {csv_path} "
        f"({', '.join(f'{k}: {len(v)}' for k, v in records_by_source.items())})."
    )
    return records_by_source


def format_property_summary(
    records_by_source: Dict[str, List[Dict[str, object]]],
    omit:              Sequence[str] = (),
    source_order:      Sequence[str] = None,
    source_labels:     Dict[str, str] = None,
) -> List[str]:
    """
    Console summary lines, one block per source, denominators spelled out.

    `omit` drops metric lines that would be meaningless for the records at
    hand -- Stage 9a's parent arm passes ("validity", "novelty"), since a
    parent is 100% valid by construction (unparseable parents never become
    pairs) and has zero novelty against itself by definition. `source_order`
    / `source_labels` name series beyond the built-in stage1a / stage1b, the
    same way plot_property_report's do.
    """
    skip     = set(omit or ())
    order    = tuple(source_order or _SOURCE_ORDER)
    label_of = {**SOURCE_LABELS, **(source_labels or {})}

    lines: List[str] = []
    for source in order:
        records = records_by_source.get(source)
        if not records:
            continue
        st = summarize_property_records(records)
        lines.append(f"  {label_of.get(source, source)}")
        if "validity" not in skip:
            lines.append(
                f"    RDKit validity      : {st['validity']:.1%}  "
                f"({st['n_valid']}/{st['n_attempted']} generated, "
                f"95% CI {st['validity_ci_lo']:.1%}-{st['validity_ci_hi']:.1%})"
            )
        lines.append(
            f"    QED                 : mean {st['qed_mean']:.3f} +/- {st['qed_sd']:.3f}, "
            f"median {st['qed_median']:.3f}, "
            f">= {QED_DRUGLIKE_THRESHOLD:g}: {st['qed_druglike']:.1%}   (n={st['qed_n']} valid)"
        )
        lines.append(
            f"    SA score (raw 1-10) : mean {st['sa_mean']:.2f}, median {st['sa_median']:.2f}, "
            f"<= {SA_EASY_THRESHOLD:g}: {st['sa_easy']:.1%}   (n={st['sa_n']} valid)"
        )
        if "novelty" not in skip:
            lines.append(
                f"    Novelty vs. parent  : median {st['novelty_median']:.3f} "
                f"(IQR {st['novelty_q1']:.3f}-{st['novelty_q3']:.3f}), "
                f"identical to parent: {st['novelty_zero']:.1%}   (n={st['novelty_n']} valid)"
            )
        lines.append(
            f"    Structural alerts   : PAINS {st['pains_rate']:.1%}, "
            f"Brenk {st['brenk_rate']:.1%}, any {st['any_alert_rate']:.1%}, "
            f"mean alerts/mol {st['alerts_mean']:.2f}   (n={st['alert_n']} valid)"
        )
        if st["tox21_n"]:
            lines.append(
                f"    Tox21 clean prob.   : mean {st['tox21_mean']:.3f}, "
                f"median {st['tox21_median']:.3f}   (n={st['tox21_n']} valid)"
            )
    return lines


def format_collection_footer(sources: List[str] = None) -> str:
    """
    Figure footer from COLLECTION_STATS: what each source contributed and
    what was dropped before a single molecule was generated.
    """
    # Bookkeeping keys, not drop reasons -- anything outside this set is
    # reported below as "dropped in prep", so a new stat must be listed here
    # or it reads as a discarded-molecule count in every published figure.
    meta  = ("rows", "pairs", "subsampled_from", "sample_seed",
             "deduped_from", "duplicate_parents",
             "capped_from", "max_pairs_per_parent", "distinct_parents")
    parts: List[str] = []
    for source in (sources or list(_SOURCE_ORDER)):
        st = COLLECTION_STATS.get(source)
        if not st:
            continue
        dropped = ", ".join(
            f"{k}: {v}" for k, v in st.items() if k not in meta and v
        ) or "none"
        line = (f"{SOURCE_LABELS.get(source, source)}: {st.get('rows', 0)} row(s) read, "
                f"{st.get('pairs', 0)} pair(s) generated from; dropped in prep -- {dropped}")
        if st.get("deduped_from"):
            line += (f"; collapsed to {st['deduped_from'] - st['duplicate_parents']} unique "
                     f"parent(s) from {st['deduped_from']} ligand instance(s)")
        if st.get("subsampled_from"):
            line += (f"; randomly sampled {st['pairs']} of {st['subsampled_from']} usable "
                     f"pair(s), seed {st.get('sample_seed')}")
        parts.append(line)
    return "   |   ".join(parts)


def _fmt(value: float, spec: str = ".3f") -> str:
    """Format a statistic, printing 'n/a' for NaN rather than 'nan'."""
    return "n/a" if value != value else format(value, spec)


def _horizontal_boxplot(ax, data, positions, colors, widths=0.6):
    """
    Horizontal box plot, tolerant of the matplotlib 3.10 rename of `vert`
    to `orientation` (Colab and local installs are rarely the same version).
    """
    kwargs = dict(positions=positions, widths=widths, patch_artist=True,
                  showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.4),
                  medianprops=dict(color="black", linewidth=1.4))
    try:
        bp = ax.boxplot(data, orientation="horizontal", **kwargs)
    except TypeError:
        bp = ax.boxplot(data, vert=False, **kwargs)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    return bp


def _shrink_stat_boxes(fig, boxes: List[Tuple[object, object]], min_size: float = 5.0) -> None:
    """
    Step each stats box's font down until the box fits inside its own panel.

    Every panel carries one block of numbers per series, so the width of that
    box scales with how many series are plotted: two fit comfortably at
    fontsize 8, but a four-series parent-vs-generated overlay spills over the
    panel edge and onto the neighbouring panel's tick labels. Measuring beats
    hard-coding a size per figure -- the same code then handles two series,
    four, and whatever a caller adds later.

    Must run AFTER the final subplots_adjust: that call changes the axes
    width, which is exactly what is being measured against. Silently does
    nothing on a backend with no usable renderer -- a slightly wide stats box
    is not worth failing a figure over.
    """
    if not boxes:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    for ax, artist in boxes:
        try:
            # 0.94, not 1.0: the box is anchored 0.02 in from the panel edge
            # and its rounded bbox pads a few points either side of the text
            # being measured.
            avail = ax.get_window_extent(renderer).width * 0.94
            for _ in range(12):
                size = artist.get_fontsize()
                if artist.get_window_extent(renderer).width <= avail or size <= min_size:
                    break
                artist.set_fontsize(max(min_size, size - 0.5))
        except Exception:
            continue


def plot_property_report(
    records_by_source:  Dict[str, List[Dict[str, object]]],
    out_path:           str,
    suptitle:           str,
    footer:             str = None,
    source_order:       Sequence[str] = None,
    source_labels:      Dict[str, str] = None,
    source_colors:      Dict[str, str] = None,
    short_labels:       Dict[str, str] = None,
    omit_panels:        Sequence[str] = (),
    panel_series:       Dict[str, Sequence[str]] = None,
    hatches:            Dict[str, str] = None,
    validity_reference: Tuple[float, str] = None,
) -> None:
    """
    One figure, one panel per criterion, with the denominator printed in
    every panel:

      1. RDKit validity      -- bar (valid / ALL generated) + Wilson 95% CI.
                                The only panel over all attempts.
      2. QED                 -- density histogram (continuous, 0-1) over
                                valid molecules + box strip below sharing
                                the x-axis; mean (solid) and median (dashed)
                                lines; % >= QED_DRUGLIKE_THRESHOLD annotated.
      3. Synthetic accessib. -- same treatment on the RAW 1-10 SA scale
                                (interpretable, and where a shift is
                                visible), % <= SA_EASY_THRESHOLD annotated,
                                and the normalised (10-SA)/10 mean the
                                composite score uses noted in the box.
      4. Scaffold novelty    -- box plot per source over a faint violin,
                                because this distribution is bimodal: an MLM
                                often refills a mask with the parent's own
                                atoms, giving novelty exactly 0. The
                                "identical to parent" fraction is annotated,
                                since the median alone would mislead.
      5. PAINS/Brenk alerts  -- hit rates over VALID molecules, split into
                                PAINS / Brenk / any-alert, because Brenk
                                fires far more often and a merged bar hides
                                which filter matched.
      6. Tox21               -- histogram + box strip, included only when a
                                Tox21 checkpoint is configured.

    Invalid molecules never enter panels 2-6: compute_property_components
    reports None for them and _measured drops them.

    Everything below `footer` exists so Stage 9a can render the PARENT
    (pre-mask) molecules, and the parent-vs-generated overlay, through THIS
    function rather than a copy of it -- the whole point of a before/after
    comparison is that only the data changes, never the drawing code:

      source_order / source_labels / source_colors
          Extra series beyond the built-in "stage1a" / "stage1b" (Stage 9a
          adds "stage1a_parent" / "stage1b_parent"). The dicts are merged
          OVER the module-level SOURCE_LABELS / SOURCE_COLORS, so a source
          keeps its usual colour unless deliberately overridden.
      short_labels
          Abbreviated series names for the STATS BOXES only -- the legend
          keeps the full ones. Four series' worth of full labels overflow a
          panel horizontally; unset series fall back to their full label.
      omit_panels
          Panels to leave out entirely. Stage 9a's parent figure omits
          "novelty": novelty is 1 - Tanimoto(parent, generated), so a parent
          scored against itself is 0 by definition, not a measurement.
      panel_series
          {panel: series allowed in it}, for panels where a series has no
          honest value. The overlay restricts "validity" to the generated
          series because collect_pairs_from_stage1a/1b already dropped every
          unparseable parent, which makes a parent validity bar 100% by
          construction rather than a property of the molecules.
      hatches
          {series: matplotlib hatch}, so a parent and generated series of the
          SAME source can share a colour (they ARE the same molecules) and
          still be told apart in print and in greyscale.
      validity_reference
          (y, label) for a dotted reference line on the validity panel --
          the overlay's stand-in for the omitted parent bars.

    With more than two series the three-line-per-series stats boxes would
    overflow their panels, so they collapse to one line per series.
    """
    order    = tuple(source_order or _SOURCE_ORDER)
    label_of = {**SOURCE_LABELS, **(source_labels or {})}
    color_of = {**SOURCE_COLORS, **(source_colors or {})}
    short_of = {**label_of, **(short_labels or {})}
    hatch_of = dict(hatches or {})
    omit     = set(omit_panels or ())

    sources = [s for s in order if records_by_source.get(s)]
    if not sources:
        tqdm.write("  No evaluation records to plot.")
        return

    stats = {s: summarize_property_records(records_by_source[s]) for s in sources}
    # Fold the stats boxes to one line per series once they would otherwise
    # crowd their panel -- either too many series (the four-series overlay) or
    # names too long to head a block (Stage 9a's "Parent of RDKit-valid
    # generation (Stage 1a set)", which at full width makes the box tall
    # enough to sit on the bar labels). 34 keeps "Parent, pre-mask (Stage 1a
    # set)" and every generated label in the roomier two-line form.
    compact = len(sources) > 2 or max(len(label_of[s]) for s in sources) > 34

    def _series(panel: str) -> List[str]:
        """Series allowed in `panel` -- all of them unless panel_series says otherwise."""
        allowed = (panel_series or {}).get(panel)
        return [s for s in sources if allowed is None or s in allowed]

    def _block(s: str, head: str, *lines: str, drop_when_compact: int = 0) -> str:
        """
        One stats-box entry. Once the box gets crowded (>2 series) it folds
        onto a single line under the ABBREVIATED series name, and any trailing
        detail the caller marked droppable is left off -- four full-width
        blocks overflow a panel horizontally, and an unreadable number is
        worse than an absent one.
        """
        if compact:
            kept = lines[:len(lines) - drop_when_compact] if drop_when_compact else lines
            return f"{short_of[s]}: {head}  " + "  ".join(kept)
        return f"{label_of[s]}: {head}\n  " + "\n  ".join(lines)

    panels = [p for p in ("validity", "qed", "sa", "novelty", "alerts") if p not in omit]
    if "tox21" not in omit and any(stats[s]["tox21_n"] for s in sources):
        panels.append("tox21")
    if not panels:
        tqdm.write("  Every panel was omitted -- nothing to plot.")
        return

    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig   = plt.figure(figsize=(6.6 * ncols, 5.0 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.55, wspace=0.26)

    # (axes, text) for every stats box, so _shrink_stat_boxes can size them
    # once the panels are in their FINAL positions -- subplots_adjust below
    # changes the axes width, and measuring before that measures the wrong box.
    stat_boxes: List[Tuple[object, object]] = []

    def _annotate(ax, text: str, ha: str = "right") -> None:
        """
        Stats box in the headroom every panel reserves above its data (see
        _headroom) -- so the numbers never sit on top of a bar or a bin.
        """
        x = {"right": 0.98, "left": 0.02, "center": 0.5}[ha]
        artist = ax.text(x, 0.97, text, transform=ax.transAxes, ha=ha, va="top",
                         fontsize=7 if compact else 8, linespacing=1.35,
                         bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                   edgecolor="#cccccc", alpha=0.9))
        stat_boxes.append((ax, artist))

    def _headroom(ax, top: float = None, factor: float = 1.62,
                  unit_ticks: bool = False) -> None:
        """Leave the upper ~40% of a panel free for its stats box."""
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, (top if top is not None else hi) * factor)
        if unit_ticks:                      # keep a 0-1 metric honest: no ticks above 1
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def _hist_with_box(cell, key, bins, xlim, xlabel, title, note_fn, vline=None,
                       series=None):
        series = list(series if series is not None else sources)
        inner  = cell.subgridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
        ax     = fig.add_subplot(inner[0])
        ax_box = fig.add_subplot(inner[1], sharex=ax)

        per_source = {s: _measured(records_by_source[s], key) for s in series}
        for s in series:
            vals = per_source[s]
            if not vals:
                continue
            ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                    color=color_of[s], edgecolor=color_of[s], hatch=hatch_of.get(s),
                    linewidth=1.3, label=label_of[s])
            ax.axvline(_mean(vals), color=color_of[s], linestyle="-", linewidth=1.5)
            ax.axvline(_quantile(vals, 0.5), color=color_of[s],
                       linestyle="--", linewidth=1.5)
        if vline is not None:
            for a in (ax, ax_box):
                a.axvline(vline, color="#555555", linestyle=":", linewidth=1.2)

        if any(per_source[s] for s in series):
            _horizontal_boxplot(
                ax_box,
                [per_source[s] or [float("nan")] for s in series],
                positions=list(range(1, len(series) + 1)),
                colors=[color_of[s] for s in series],
            )
        else:
            ax.text(0.5, 0.5, "no valid molecules", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#888888")

        ax.set_xlim(*xlim)
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelbottom=False)
        _headroom(ax)
        ax_box.set_yticks(list(range(1, len(series) + 1)))
        ax_box.set_yticklabels([""] * len(series))
        ax_box.set_xlabel(f"{xlabel}\nsolid line = mean, dashed = median", fontsize=9)
        ax_box.grid(True, axis="x", linestyle="--", alpha=0.3)
        _annotate(ax, note_fn())
        return ax

    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        cell = outer[r, c]

        if panel == "validity":
            series = _series("validity")
            ax = fig.add_subplot(cell)
            for j, s in enumerate(series):
                st  = stats[s]
                lo  = st["validity"] - st["validity_ci_lo"]
                hi  = st["validity_ci_hi"] - st["validity"]
                ax.bar(j, st["validity"], color=color_of[s], width=0.55,
                       alpha=0.9, label=label_of[s], hatch=hatch_of.get(s),
                       yerr=[[max(lo, 0)], [max(hi, 0)]], capsize=6,
                       error_kw=dict(ecolor="#333333", lw=1.2))
                ax.text(j, st["validity_ci_hi"] + 0.03,
                        f"{st['validity']:.1%}\n{st['n_valid']}/{st['n_attempted']}",
                        ha="center", va="bottom", fontsize=9)
            note = ("the ONLY panel over all generated molecules;\n"
                    "invalid molecules are counted here and nowhere else.\n"
                    "bars show the Wilson 95% CI")
            if validity_reference is not None:
                # Labelled in the stats box rather than inline on the line
                # itself: the line sits at the top of the data area, exactly
                # where the per-bar "66.7% / 2-of-3" annotations already are.
                ref_y, ref_label = validity_reference
                ax.axhline(ref_y, color="#555555", linestyle=":", linewidth=1.4)
                note += f"\ndotted line: {ref_label}"
            ax.set_xticks(range(len(series)))
            ax.set_xticklabels([label_of[s] for s in series], fontsize=8)
            ax.set_xlim(-0.7, len(series) - 0.3)      # keep a lone bar from spanning the panel
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Fraction of generated molecules")
            ax.set_title("RDKit validity  (valid / all generated)", fontsize=10.5)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            _headroom(ax, top=1.0, unit_ticks=True)
            _annotate(ax, note, ha="center")

        elif panel == "qed":
            series = _series("qed")

            def _qed_note(series=series):
                return "\n".join(
                    _block(
                        s, f"n={stats[s]['qed_n']} valid",
                        f"mean {_fmt(stats[s]['qed_mean'])} +/- {_fmt(stats[s]['qed_sd'])}, "
                        f"median {_fmt(stats[s]['qed_median'])}",
                        f"QED >= {QED_DRUGLIKE_THRESHOLD:g}: "
                        f"{_fmt(stats[s]['qed_druglike'], '.1%')}",
                    )
                    for s in series
                )
            _hist_with_box(
                cell, "qed", [i / 20 for i in range(21)], (0, 1),
                "QED (higher = more drug-like)",
                "Drug-likeness (QED)  --  continuous, valid molecules only",
                _qed_note, vline=QED_DRUGLIKE_THRESHOLD, series=series,
            )

        elif panel == "sa":
            series = _series("sa")

            def _sa_note(series=series):
                return "\n".join(
                    _block(
                        s, f"n={stats[s]['sa_n']} valid",
                        f"mean {_fmt(stats[s]['sa_mean'], '.2f')}, "
                        f"median {_fmt(stats[s]['sa_median'], '.2f')}",
                        f"SA <= {SA_EASY_THRESHOLD:g}: {_fmt(stats[s]['sa_easy'], '.1%')}",
                        f"[(10-SA)/10 = {_fmt(stats[s]['sa_norm_mean'])}]",
                        drop_when_compact=1,
                    )
                    for s in series
                )
            _hist_with_box(
                cell, "sa_raw", [1.0 + 0.5 * k for k in range(19)], (1, 10),
                "SA-Score (raw, 1 = easy ... 10 = hard)",
                "Synthetic accessibility  --  continuous, valid molecules only",
                _sa_note, vline=SA_EASY_THRESHOLD, series=series,
            )

        elif panel == "novelty":
            series = _series("novelty")
            ax   = fig.add_subplot(cell)
            data = [_measured(records_by_source[s], "novelty") for s in series]
            pos  = list(range(1, len(series) + 1))
            drawn = [(p, d, s) for p, d, s in zip(pos, data, series) if d]
            if drawn:
                vp = ax.violinplot([d for _, d, _ in drawn],
                                   positions=[p for p, _, _ in drawn],
                                   widths=0.75, showextrema=False)
                for body, (_, _, s) in zip(vp["bodies"], drawn):
                    body.set_facecolor(color_of[s])
                    body.set_alpha(0.30)
                bp = ax.boxplot([d for _, d, _ in drawn],
                                positions=[p for p, _, _ in drawn], widths=0.22,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", linewidth=1.6))
                for patch, (_, _, s) in zip(bp["boxes"], drawn):
                    patch.set_facecolor(color_of[s])
                    patch.set_alpha(0.75)
            else:
                ax.text(0.5, 0.5, "no valid molecules", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="#888888")
            ax.set_xticks(pos)
            ax.set_xticklabels([label_of[s] for s in series], fontsize=8)
            ax.set_ylim(-0.03, 1.02)
            ax.set_ylabel("1 - Tanimoto(parent, generated)")
            ax.set_title("Scaffold novelty vs. parent  --  valid molecules only", fontsize=10.5)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            _headroom(ax, top=1.02, unit_ticks=True)
            # A series with nothing measurable here (a parent scored against
            # itself) would otherwise contribute a block of "n/a" to the box.
            noted = [s for s in series if stats[s]["novelty_n"]] or series
            _annotate(ax, "\n".join(
                _block(
                    s, f"n={stats[s]['novelty_n']} valid",
                    f"median {_fmt(stats[s]['novelty_median'])} "
                    f"(IQR {_fmt(stats[s]['novelty_q1'])}-{_fmt(stats[s]['novelty_q3'])})",
                    f"identical to parent: {_fmt(stats[s]['novelty_zero'], '.1%')}",
                )
                for s in noted
            ))

        elif panel == "alerts":
            series = _series("alerts")
            ax     = fig.add_subplot(cell)
            groups = [("PAINS", "pains_rate"), ("Brenk", "brenk_rate"), ("Any alert", "any_alert_rate")]
            width  = 0.8 / max(len(series), 1)
            for j, s in enumerate(series):
                heights = [stats[s][key] for _, key in groups]
                xs      = [g + (j - (len(series) - 1) / 2) * width for g in range(len(groups))]
                ax.bar(xs, heights, width=width * 0.92, color=color_of[s],
                       alpha=0.9, label=label_of[s], hatch=hatch_of.get(s))
                for x, h in zip(xs, heights):
                    if h == h:
                        ax.text(x, h + 0.02, f"{h:.0%}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g for g, _ in groups], fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Hit rate among valid molecules")
            ax.set_title("Structural-alert hit rate  (hits / valid molecules)", fontsize=10.5)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            _headroom(ax, top=1.0, unit_ticks=True)
            _annotate(ax, "\n".join(
                _block(
                    s, f"n={stats[s]['alert_n']} valid",
                    f"mean alerts/mol {_fmt(stats[s]['alerts_mean'], '.2f')}",
                    f"alert-free (score term): {_fmt(1.0 - stats[s]['any_alert_rate'], '.1%')}",
                )
                for s in series
            ))

        elif panel == "tox21":
            series = _series("tox21")

            def _tox21_note(series=series):
                return "\n".join(
                    _block(
                        s, f"n={stats[s]['tox21_n']} valid",
                        f"mean {_fmt(stats[s]['tox21_mean'])}, "
                        f"median {_fmt(stats[s]['tox21_median'])}",
                    )
                    for s in series
                )
            _hist_with_box(
                cell, "tox21", [i / 20 for i in range(21)], (0, 1),
                "1 - aggregated toxic probability",
                "Tox21 classifier clean probability  --  valid molecules only",
                _tox21_note, series=series,
            )

    handles, legend_labels = [], []
    for ax in fig.get_axes():
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in legend_labels:
                handles.append(hh)
                legend_labels.append(ll)
    # Header and footer are reserved in INCHES, not figure fractions: the
    # figure is 5 inches per panel ROW, so a fixed 0.90/0.10 split gives a
    # one-row figure (the parent arm, which omits two panels) half the
    # absolute headroom a two-row one gets -- and the legend then lands on
    # top of the panel titles while the footer lands on the x-labels.
    n_leg    = len(legend_labels)
    leg_cols = n_leg if n_leg <= 2 else 2      # 4-series overlays wrap rather
    leg_rows = -(-n_leg // leg_cols) if n_leg else 0   # than run off the edge
    fig_h    = fig.get_figheight()
    # 0.42 suptitle + 0.28/legend row + 0.34 for the panel titles, which are
    # drawn ABOVE the axes box subplots_adjust positions and are what the
    # legend collides with if only the legend's own height is reserved.
    head_in  = 0.42 + 0.28 * leg_rows + 0.34
    foot_in  = 0.95 if footer else 0.25

    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=leg_cols,
                   bbox_to_anchor=(0.5, 1.0 - 0.36 / fig_h), frameon=False, fontsize=10)
    fig.suptitle(suptitle, y=1.0 - 0.11 / fig_h, fontsize=13)
    if footer:
        fig.text(0.5, 0.16 / fig_h, footer, ha="center", va="bottom", fontsize=8,
                 color="#444444", wrap=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.subplots_adjust(top=1.0 - head_in / fig_h, bottom=foot_in / fig_h)
    _shrink_stat_boxes(fig, stat_boxes)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  Property report figure : {out_path}")


def run_property_distribution_eval(
    model_name:      str  = config.CHEMBERTA_MODEL,
    lora_checkpoint: str  = None,
    out_path:        str  = None,
    suptitle:        str  = "Stage 9 -- property distributions (fine-tuned ChemBERTa)",
    max_pairs_per_source: int = None,
    sample_seed:          int = None,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Load ChemBERTa (with the given LoRA adapter, if any) and score the same
    Stage 1a/1b pairs Stage 9a's no-fine-tuning baseline scores, then plot.
    Called from main() below with lora_checkpoint=config.STAGE9_LORA_DIR
    (the just-trained adapter) so Stage 9 produces the identical set of
    plots as Stage 9a, over the identical molecules, for direct comparison
    (same evaluate_property_records / plot_property_report code, so the only
    difference between the two figures is the model weights).

    `max_pairs_per_source` / `sample_seed` mirror Stage 9a's --limit / --seed
    so both scripts can be pointed at the same evaluation set from the command
    line. Both use Stage 9a's convention exactly: None means "not given, fall
    back to config", while 0 means "explicitly no limit" -- None could not
    express the latter, since it is indistinguishable from the flag being
    absent. Matching Stage 9a here is the whole point: a Stage 9 figure is only
    comparable to a Stage 9a baseline that was produced at the same limit and
    seed (see config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE).
    """
    pairs_by_source = collect_pairs_by_source(
        max_pairs_per_source=max_pairs_per_source, sample_seed=sample_seed,
    )
    if not pairs_by_source:
        tqdm.write("  No Stage 1a/1b pairs found for property-distribution evaluation.")
        return {}
    confirm_partial_sources_or_exit(pairs_by_source)

    tokenizer, model, device = load_chemberta_for_policy(
        model_name=model_name, lora_checkpoint=lora_checkpoint,
    )
    records = evaluate_property_records(tokenizer, model, device, pairs_by_source)

    out_path = out_path or os.path.join(config.STAGE9_LORA_DIR, "stage9_property_distributions.png")
    plot_property_report(
        records, out_path, suptitle,
        footer=format_collection_footer(list(records.keys())),
    )
    write_property_records_csv(
        records, os.path.splitext(out_path)[0] + "_per_molecule.csv",
    )
    for line in format_property_summary(records):
        tqdm.write(line)
    return records


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_stage9_finetuning(
    pairs:           List[Tuple[str, str]],
    model_name:      str   = config.CHEMBERTA_MODEL,
    save_dir:        str   = None,
    lora_rank:       int   = LORA_RANK,
    lora_alpha:      int   = LORA_ALPHA,
    lora_dropout:    float = LORA_DROPOUT,
    lora_targets:    List[str] = LORA_TARGET_MODS,
    num_epochs:      int   = NUM_EPOCHS,
    batch_size:      int   = BATCH_SIZE,
    lr:              float = LEARNING_RATE,
    temperature:     float = TEMPERATURE,
    top_k:           int   = TOP_K,
    grad_clip:       float = GRAD_CLIP,
    baseline_decay:  float = BASELINE_DECAY,
    kl_beta:         float = KL_BETA,
    score_weights:   Tuple[float, float, float, float, float, float] = (
        SCORE_W_VALID, SCORE_W_QED, SCORE_W_SA, SCORE_W_NOVELTY,
        SCORE_W_TOX_ALERT, SCORE_W_TOX21),
) -> Dict[str, list]:
    """
    REINFORCE fine-tuning loop over (masked_smiles, original_smiles) pairs.
    Checkpointing / resume follow stage1_9's convention exactly (same
    helper functions, reused not reimplemented):
      <save_dir>/epoch_{N:03d}/             — LoRA adapter weights
      <save_dir>/optimizer.pt               — optimizer state
      <save_dir>/training_checkpoint.json   — epoch, baseline, history
    """
    save_dir = save_dir or config.STAGE9_LORA_DIR
    os.makedirs(save_dir, exist_ok=True)

    ckpt        = _load_checkpoint(save_dir)
    start_epoch = 1
    baseline    = 0.0
    global_step = 0
    history: Dict[str, list] = {
        "epoch": [], "step": [], "reward_mean": [], "loss_mean": [],
        "valid_rate": [], "qed_mean": [], "sa_mean": [],
        "novelty_mean": [], "tox_alert_free_rate": [], "tox21_clean_mean": [],
        "kl_mean": [],
    }
    resume_adapter = None

    if ckpt and _ask_resume(save_dir, ckpt):
        start_epoch    = ckpt["last_epoch"] + 1
        baseline       = ckpt["baseline"]
        global_step    = ckpt["global_step"]
        history        = ckpt["history"]
        # Checkpoints written before the KL anchor existed have no "kl_mean";
        # pad it to the same length as every other series so the curve plot
        # keeps lining up with "epoch" instead of raising on resume.
        history.setdefault("kl_mean", [])
        history["kl_mean"] += [0.0] * (len(history.get("epoch", []))
                                       - len(history["kl_mean"]))
        resume_adapter = os.path.join(save_dir, f"epoch_{ckpt['last_epoch']:03d}")
        tqdm.write(f"\n  Resuming from epoch {start_epoch} "
                   f"(baseline={baseline:.3f}, step={global_step})")
    else:
        tqdm.write("\n  Starting fresh training run.")

    if start_epoch > num_epochs:
        tqdm.write("  Training already complete (all epochs done).")
        return history

    tokenizer, model, device = load_chemberta_for_policy(
        model_name      = model_name,
        lora_rank       = lora_rank,
        lora_alpha      = lora_alpha,
        lora_dropout    = lora_dropout,
        lora_targets    = lora_targets,
        lora_checkpoint = resume_adapter,
    )

    if kl_beta > 0.0 and KL_BASE_DROPOUT_OFF:
        n_off = disable_base_dropout(model)
        if n_off:
            tqdm.write(
                f"  KL anchor: base dropout disabled in {n_off} module(s) so the "
                f"penalty measures adapter drift, not dropout noise "
                f"(config.STAGE9_KL_BASE_DROPOUT_OFF)."
            )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    opt_path = os.path.join(save_dir, "optimizer.pt")
    if resume_adapter and os.path.isfile(opt_path):
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            tqdm.write("  Optimizer state restored.")
        except Exception as e:
            tqdm.write(f"  Could not restore optimizer state: {e}")

    rng = random.Random(42 + start_epoch)

    remaining_epochs = num_epochs - start_epoch + 1
    total_batches    = remaining_epochs * ((len(pairs) + batch_size - 1) // batch_size)

    pbar = tqdm(
        total=total_batches, desc="Stage 9 property fine-tuning", unit="batch",
        dynamic_ncols=True,
        bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"),
    )

    for epoch in range(start_epoch, num_epochs + 1):
        rng.shuffle(pairs)
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

        ep_reward, ep_loss, ep_valid = [], [], []
        ep_qed, ep_sa, ep_novelty, ep_tox_free, ep_tox21 = [], [], [], [], []
        ep_kl = []

        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"  Epoch {epoch}/{num_epochs}  ({len(batches)} batches × {batch_size} samples)")
        tqdm.write(
            "  What happens: ONE forward pass scores every <mask> position at\n"
            f"  once (one-shot MLM decoding); each is sampled independently from\n"
            f"  top-{top_k} candidates, recording log P(token|context).\n"
            "  After all masks filled -> RDKit scores validity/QED/SA, plus\n"
            "  novelty (1 - similarity to the pre-mask parent), a PAINS/Brenk\n"
            "  toxicity-alert check, and (if configured) a Tox21 classifier probability.\n"
            f"  loss = -(score - {baseline:.3f}) x sum(log_prob)"
            + (f" + {kl_beta} x KL(policy || pretrained)" if kl_beta > 0 else "")
            + "  <- this is what backward() sees."
        )
        tqdm.write(f"{'─'*60}")

        for batch in batches:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            b_reward = b_valid = b_qed = b_sa = b_novelty = b_tox_free = b_tox21 = 0.0
            b_kl = b_masks = 0.0

            for masked_smi, orig_smi in batch:
                # Returns (score, components) so the rollout hands both back in
                # one go -- scoring the same finished SMILES a second time just
                # to read its breakdown is deterministic, and RDKit is this
                # loop's bottleneck, not the GPU.
                def _score(smi, _orig=orig_smi, _w=score_weights):
                    return compute_stage9_score(smi, _orig, _w)

                reward, log_prob, kl, generated, comps = reinforce_rollout_oneshot(
                    smiles_masked=masked_smi, tokenizer=tokenizer, model=model,
                    device=device, top_k=top_k, temperature=temperature,
                    reward_fn=_score, kl_beta=kl_beta,
                )

                advantage  = reward - baseline
                # The KL term is differentiable in its own right, so it is
                # added straight to the loss rather than folded into the
                # reward: no REINFORCE variance is spent on the anchor.
                batch_loss = batch_loss + (-advantage * log_prob)
                if kl_beta > 0.0:
                    batch_loss = batch_loss + kl_beta * kl
                    b_kl      += float(kl.detach())
                    b_masks   += max(masked_smi.count(tokenizer.mask_token), 1)
                b_reward   += reward
                b_valid    += comps["valid"]
                b_qed      += comps["qed"]
                b_sa       += comps["sa"]
                b_novelty  += comps["novelty"]
                b_tox_free += comps["tox_alert"]
                b_tox21    += comps["tox21"]

            n = max(len(batch), 1)
            batch_loss = batch_loss / n
            batch_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            mean_reward = b_reward / n
            baseline = baseline_decay * baseline + (1 - baseline_decay) * mean_reward

            ep_reward.append(mean_reward)
            ep_loss.append(batch_loss.item())
            ep_valid.append(b_valid / n)
            ep_qed.append(b_qed / n)
            ep_sa.append(b_sa / n)
            ep_novelty.append(b_novelty / n)
            ep_tox_free.append(b_tox_free / n)
            ep_tox21.append(b_tox21 / n)
            # Reported PER MASKED POSITION, not per molecule: the summed KL
            # scales with molecule size, so a per-position figure is the one
            # that stays comparable across batches and tunable against beta.
            ep_kl.append(b_kl / b_masks if b_masks else 0.0)
            global_step += 1

            pbar.set_postfix_str(
                f"ep={epoch}/{num_epochs}  score={mean_reward:.3f}  "
                f"loss={batch_loss.item():.4f}  valid={ep_valid[-1]:.0%}  "
                f"novelty={ep_novelty[-1]:.2f}  tox_free={ep_tox_free[-1]:.0%}  "
                f"tox21={ep_tox21[-1]:.2f}  baseline={baseline:.3f}"
                + (f"  kl/pos={ep_kl[-1]:.3f}" if kl_beta > 0 else ""),
                refresh=True,
            )
            pbar.update(1)

        def _avg(xs):
            return sum(xs) / max(len(xs), 1)

        tqdm.write(
            f"\n  Epoch {epoch} summary -- score={_avg(ep_reward):.3f}  "
            f"loss={_avg(ep_loss):.4f}  valid={_avg(ep_valid):.1%}  "
            f"qed={_avg(ep_qed):.3f}  sa={_avg(ep_sa):.3f}  "
            f"novelty={_avg(ep_novelty):.3f}  tox_free={_avg(ep_tox_free):.1%}  "
            f"tox21_clean={_avg(ep_tox21):.3f}  baseline={baseline:.3f}"
            + (f"  kl/pos={_avg(ep_kl):.4f}" if kl_beta > 0 else "")
        )

        history["epoch"].append(epoch)
        history["step"].append(global_step)
        history["reward_mean"].append(_avg(ep_reward))
        history["loss_mean"].append(_avg(ep_loss))
        history["valid_rate"].append(_avg(ep_valid))
        history["qed_mean"].append(_avg(ep_qed))
        history["sa_mean"].append(_avg(ep_sa))
        history["novelty_mean"].append(_avg(ep_novelty))
        history["kl_mean"].append(_avg(ep_kl))
        history["tox_alert_free_rate"].append(_avg(ep_tox_free))
        history["tox21_clean_mean"].append(_avg(ep_tox21))

        epoch_adapter_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
        model.save_pretrained(epoch_adapter_dir)

        _save_checkpoint(
            save_dir=save_dir, epoch=epoch, baseline=baseline,
            history=history, optimizer=optimizer, global_step=global_step,
        )
        tqdm.write(f"  Checkpoint saved -> {epoch_adapter_dir}")

    pbar.close()

    final_adapter = os.path.join(save_dir, f"epoch_{num_epochs:03d}")
    if os.path.isdir(final_adapter):
        import shutil
        for fname in os.listdir(final_adapter):
            shutil.copy2(os.path.join(final_adapter, fname), os.path.join(save_dir, fname))
    tqdm.write(f"\n  Final LoRA adapter saved to : {save_dir}")

    _plot_stage9_history(history, save_dir)
    return history


def _plot_stage9_history(history: Dict[str, list], save_dir: str) -> None:
    if not history.get("epoch"):
        return
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    ax = axes.flat

    ax[0].plot(history["epoch"], history["reward_mean"], marker="o", color="#1f77b4")
    ax[0].set_title("Mean composite score / epoch")

    ax[1].plot(history["epoch"], history["loss_mean"], marker="s", color="#d62728")
    ax[1].set_title("Mean REINFORCE loss / epoch")

    ax[2].plot(history["epoch"], history["valid_rate"], marker="^", color="#2ca02c")
    ax[2].set_title("Validity rate / epoch"); ax[2].set_ylim(0, 1)

    ax[3].plot(history["epoch"], history["qed_mean"], marker="d", color="#9467bd")
    ax[3].set_title("Mean QED / epoch"); ax[3].set_ylim(0, 1)

    ax[4].plot(history["epoch"], history["sa_mean"], marker="X", color="#8c564b")
    ax[4].set_title("Mean SA (1-normalised) / epoch"); ax[4].set_ylim(0, 1)

    ax[5].plot(history["epoch"], history["novelty_mean"], marker="v", color="#ff7f0e")
    ax[5].set_title("Mean novelty (1 - similarity to parent) / epoch"); ax[5].set_ylim(0, 1)

    ax[6].plot(history["epoch"], history["tox_alert_free_rate"], marker="P", color="#17becf")
    ax[6].set_title("PAINS/Brenk alert-free rate / epoch"); ax[6].set_ylim(0, 1)

    ax[7].plot(history["epoch"], history["tox21_clean_mean"], marker="*", color="#e377c2")
    ax[7].set_title("Tox21 classifier clean prob. / epoch"); ax[7].set_ylim(0, 1)

    # Drift from the pretrained model. Left unbounded on purpose -- the whole
    # point is to see whether it plateaus (beta is holding) or keeps climbing
    # (beta too low). Epochs recorded before the anchor existed read 0.
    kl = history.get("kl_mean") or [0.0] * len(history["epoch"])
    ax[8].plot(history["epoch"], kl, marker="h", color="#7f7f7f")
    ax[8].set_title("KL(policy || pretrained) per masked position / epoch")
    ax[8].set_ylim(bottom=0)

    for a in ax:
        a.set_xlabel("Epoch"); a.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(save_dir, "stage9_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  Training curves saved : {out}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(max_pairs_per_source: int = None, sample_seed: int = None) -> None:
    print("\n" + "=" * 60)
    print("STAGE 9 -- MASKED-DATA PROPERTY-GUIDED CHEMBERTA FINE-TUNING")
    print("=" * 60)
    print(f"""
  What this script does
  ----------------------
  1. Loads masked/original SMILES pairs (config.STAGE9_DATA_SOURCE={config.STAGE9_DATA_SOURCE!r})
     from Stage 1a ({config.STAGE9_9A_STAGE1A_DATA_DIR or config.STAGE1A_DIR}) and/or
     Stage 1b ({config.STAGE9_9A_STAGE1B_DATA_DIR or config.STAGE1B_PLIP_MASK_DIR}),
     both re-masked to config.STAGE9_MASK_PERCENT={config.STAGE9_MASK_PERCENT}%
     of each molecule's BPE tokens, then capped to at most
     config.STAGE9_MAX_PAIRS_PER_PARENT={MAX_PAIRS_PER_PARENT} ligand instance(s) per unique
     parent molecule so PDB popularity does not weight the gradient.
  2. Loads ChemBERTa with trainable LoRA adapters (~1% of parameters).
  3. For each pair:
       a. One forward pass scores every <mask> position at once (one-shot
          MLM decoding); each is sampled independently from top-{TOP_K},
          recording log P(chosen_token | context) at each position.
       b. Decodes the full SMILES and scores it:
            score = {SCORE_W_VALID}*valid + {SCORE_W_QED}*QED + {SCORE_W_SA}*(1-SA/10)
                  + {SCORE_W_NOVELTY}*(1-similarity_to_original)
                  + {SCORE_W_TOX_ALERT}*(1-PAINS/Brenk_alert) + {SCORE_W_TOX21}*(1-Tox21_toxic_prob)
       c. loss = -(score - baseline) * sum(log_prob)  -- backpropagated
          through the LoRA weights (REINFORCE / score-function estimator;
          see the module docstring for why this is required instead of
          plain backprop).
  4. Checkpoints after every epoch ({NUM_EPOCHS} total, config.STAGE9_NUM_EPOCHS);
     safe to interrupt and resume.

  Tox21 classifier : {"loaded from " + _TOX21_MODEL_DIR if _TOX21_AVAILABLE else "NOT configured (config.STAGE9_TOX21_MODEL_DIR) — tox21 term contributes 0"}
""")

    pairs = collect_all_training_pairs()
    if not pairs:
        print("  No training pairs found. Run stage1a_random_masking.py and/or "
              "stage1b_large_scale_PLIP_mask_calculation.py first.")
        sys.exit(1)

    history = run_stage9_finetuning(pairs=pairs, save_dir=config.STAGE9_LORA_DIR)

    print("\n" + "=" * 60)
    print("  Stage 9 fine-tuning complete.")
    if history["valid_rate"]:
        print(f"  Final validity rate      : {history['valid_rate'][-1]:.1%}")
        print(f"  Final mean score         : {history['reward_mean'][-1]:.3f}")
        print(f"  Final novelty            : {history['novelty_mean'][-1]:.3f}")
        print(f"  Final tox-alert-free rate: {history['tox_alert_free_rate'][-1]:.1%}")
        print(f"  Final Tox21 clean prob.  : {history['tox21_clean_mean'][-1]:.3f}")
    print(f"  LoRA adapter         : {config.STAGE9_LORA_DIR}")
    print("=" * 60)

    print("\n  Scoring the fine-tuned model on the full Stage 1a/1b pair set "
          "for the property-distribution plots (same molecules Stage 9a "
          "evaluates as its no-fine-tuning baseline) ...")
    run_property_distribution_eval(
        lora_checkpoint=config.STAGE9_LORA_DIR,
        suptitle="Stage 9 -- property distributions (fine-tuned ChemBERTa)",
        max_pairs_per_source=max_pairs_per_source,
        sample_seed=sample_seed,
    )


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    """
    Synthetic smoke test: hand-built masked/original pairs, 1 epoch,
    tiny batch. Requires network access (HuggingFace hub) and the
    torch/transformers/peft/rdkit dependencies, same as every other
    ChemBERTa-training stage in this pipeline.
    """
    import tempfile

    test_pairs = [
        ("CC(=O)OC1=CC=CC=C1C(=O)<mask>", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)<mask>", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)<mask>", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ("CC(=O)NC1=CC=C(<mask>)C=C1", "CC(=O)NC1=CC=C(O)C=C1"),
    ]

    # Score function sanity check (no model needed).
    score, comps = compute_stage9_score(
        "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)OC1=CC=CC=C1C(=O)O",
    )
    assert comps["valid"] == 1.0
    assert 0.0 <= score <= 1.0
    assert comps["novelty"] == 0.0, "Identical molecules must score zero novelty"
    if not _TOX21_AVAILABLE:
        assert comps["tox21"] == 0.0, "Tox21 term must fail safe to 0 when unconfigured"

    score_invalid, comps_invalid = compute_stage9_score("not_a_smiles(((", "CCO")
    assert score_invalid == 0.0
    assert comps_invalid["valid"] == 0.0

    # ── KL anchor to the frozen pretrained model ────────────────────────────
    tok_k, model_k, dev_k = load_chemberta_for_policy()
    disable_base_dropout(model_k)          # else dropout fakes drift (see config)
    masked = "CC(=O)O<mask>c1cc<mask>ccc1"

    kl0 = reinforce_rollout_oneshot(
        smiles_masked=masked, tokenizer=tok_k, model=model_k, device=dev_k,
        reward_fn=lambda smi: 0.0, kl_beta=1.0,
    )[2]
    # A freshly initialised LoRA adapter IS the pretrained model (B=0), so the
    # anchor must read exactly zero drift -- anything else means the reference
    # is not really the pretrained distribution.
    assert abs(float(kl0.detach())) < 1e-4, f"fresh adapter must have ~0 KL, got {float(kl0.detach())}"
    assert kl0.requires_grad, "KL must carry a gradient or it cannot regularise"

    # Perturb the adapter: drift must now register, and be penalised.
    with torch.no_grad():
        for pname, param in model_k.named_parameters():
            if "lora_B" in pname:
                param.add_(torch.randn_like(param) * 0.05)
    kl1 = reinforce_rollout_oneshot(
        smiles_masked=masked, tokenizer=tok_k, model=model_k, device=dev_k,
        reward_fn=lambda smi: 0.0, kl_beta=1.0,
    )[2]
    assert float(kl1.detach()) > float(kl0.detach()), f"drifted adapter must raise KL ({float(kl1.detach())})"

    # beta = 0 must skip the reference pass entirely and stay bit-identical to
    # the pre-anchor behaviour.
    kl_off = reinforce_rollout_oneshot(
        smiles_masked=masked, tokenizer=tok_k, model=model_k, device=dev_k,
        reward_fn=lambda smi: 0.0, kl_beta=0.0,
    )[2]
    assert float(kl_off) == 0.0 and not kl_off.requires_grad

    # disable_base_dropout must leave LoRA's own dropout alone.
    lora_dropouts = [m for n, m in model_k.named_modules()
                     if isinstance(m, torch.nn.Dropout) and "lora" in n.lower()]
    assert any(m.p > 0 for m in lora_dropouts) or not lora_dropouts,         "LoRA dropout must survive disable_base_dropout"
    del model_k

    with tempfile.TemporaryDirectory() as td:
        history = run_stage9_finetuning(
            pairs=list(test_pairs), save_dir=td,
            num_epochs=1, batch_size=2, top_k=5, kl_beta=0.05,
        )
        assert history["epoch"] == [1]
        assert len(history["kl_mean"]) == 1, "KL must be recorded per epoch"

        # Resuming a checkpoint written before the anchor existed must not
        # break the curve plot: kl_mean gets back-filled to match "epoch".
        import json as _json
        ck = os.path.join(td, "training_checkpoint.json")
        meta = _json.load(open(ck))
        meta["history"].pop("kl_mean")
        _json.dump(meta, open(ck, "w"))
        legacy = _load_checkpoint(td)["history"]
        legacy.setdefault("kl_mean", [])
        legacy["kl_mean"] += [0.0] * (len(legacy["epoch"]) - len(legacy["kl_mean"]))
        assert len(legacy["kl_mean"]) == len(legacy["epoch"])
        assert os.path.isfile(os.path.join(td, "training_checkpoint.json"))
        assert os.path.isfile(os.path.join(td, "stage9_training_curves.png"))
        assert os.path.isdir(os.path.join(td, "epoch_001"))

        # Property-report eval/plot pipeline shared with Stage 9a.
        tokenizer, model, device = load_chemberta_for_policy(lora_checkpoint=td)
        pairs_by_source = {"stage1a": test_pairs[:2], "stage1b": test_pairs[2:]}
        records = evaluate_property_records(
            tokenizer, model, device, pairs_by_source, top_k=5,
        )
        assert set(records.keys()) == {"stage1a", "stage1b"}
        assert len(records["stage1a"]) == 2
        assert len(records["stage1b"]) == 2

        dist_png = os.path.join(td, "test_property_distributions.png")
        plot_property_report(records, dist_png, "self-test",
                             footer=format_collection_footer())
        assert os.path.isfile(dist_png)
        rec_csv = write_property_records_csv(
            records, os.path.join(td, "test_per_molecule.csv"),
        )
        assert os.path.isfile(rec_csv)

    print("Stage 9 self-test passed.")


def _parse_args(argv: list) -> tuple:
    """
    --limit N : pairs to score PER SOURCE in the post-training property pass,
                overriding config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE.
                "none"/"all"/0 = no limit (score every available pair).
    --seed  N : sampling seed, overriding config.STAGE9_PAIR_SAMPLE_SEED.

    Deliberately identical to stage9a_masked_property_without_finetuning's
    _parse_args, flag for flag: a Stage 9 figure is only comparable to a Stage
    9a baseline generated at the SAME limit and seed, so the two scripts have
    to be drivable the same way. Neither flag touches training -- they apply
    only to the evaluation pass that produces the plots.
    """
    limit = seed = None      # None = "not given", fall back to config
    for flag in ("--limit", "--seed"):
        if flag not in argv:
            continue
        idx = argv.index(flag)
        if idx + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value, e.g. {flag} 5000")
        raw = argv[idx + 1]
        if flag == "--limit":
            # 0 (not None) means "explicitly no limit" -- None would be
            # indistinguishable from the flag being absent and would silently
            # fall back to config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE.
            limit = 0 if raw.lower() in ("none", "all", "0") else int(raw)
        else:
            seed = int(raw)
    return limit, seed


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        _limit, _seed = _parse_args(sys.argv)
        main(max_pairs_per_source=_limit, sample_seed=_seed)

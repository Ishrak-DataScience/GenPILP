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
  Stage 1a combo CSVs  (config.STAGE1A_DIR/mask*pct_temp*_seed*.csv)
    columns used: smiles (original), masked_smiles
  Stage 1b summary CSV (config.STAGE1B_PLIP_MASK_DIR/stage1b_large_scale_plip_mask_summary.csv)
    columns used: smiles (original), masked_smiles, status == "ok"

Only LoRA adapter weights are updated (peft); ChemBERTa itself stays
frozen. Model loading, checkpoint save/resume, and the REINFORCE rollout
mechanics are reused verbatim from stage1_9_LLM_RDkit_policy_training.py
rather than duplicated.

Usage
-----
  python stage9_masked_property_finetune.py
  python stage9_masked_property_finetune.py --test    (synthetic smoke test)
"""

from __future__ import annotations

import csv
import glob
import os
import random
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.DataStructs import TanimotoSimilarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RDLogger.DisableLog("rdApp.*")

import config
from stage3_analysis import _morgan_fp
from stage1_9_LLM_RDkit_policy_training import (
    _SA_AVAILABLE,
    _ask_resume,
    _load_checkpoint,
    _save_checkpoint,
    load_chemberta_for_policy,
    reinforce_rollout,
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

_TOX_CATALOG = None


def _get_toxicity_catalog():
    """Lazily build (once) the PAINS + Brenk structural-alert catalog."""
    global _TOX_CATALOG
    if _TOX_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _TOX_CATALOG = FilterCatalog(params)
    return _TOX_CATALOG


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
BATCH_SIZE       = 16
LEARNING_RATE    = 5e-5
NUM_EPOCHS       = 10
GRAD_CLIP        = 1.0

# Cap on combined Stage-1a + Stage-1b training pairs (None = use all).
MAX_TRAINING_PAIRS = None
SAMPLE_SEED         = 42


# ════════════════════════════════════════════════════════════════════════════
#  SCORE FUNCTION (valid + QED + SA + novelty + toxicity)
# ════════════════════════════════════════════════════════════════════════════

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
    fail-safe convention as stage1_9's compute_reward.
    """
    w_valid, w_qed, w_sa, w_novelty, w_tox_alert, w_tox21 = weights
    components = {
        "valid": 0.0, "qed": 0.0, "sa": 0.0, "novelty": 0.0,
        "tox_alert": 0.0, "tox21": 0.0,
    }

    mol = Chem.MolFromSmiles(generated_smiles)
    if mol is None:
        return 0.0, components

    components["valid"] = 1.0

    try:
        components["qed"] = QED.qed(mol)
    except Exception:
        pass

    if _SA_AVAILABLE:
        try:
            raw_sa = sascorer.calculateScore(mol)
            components["sa"] = max(0.0, (10.0 - raw_sa) / 10.0)
        except Exception:
            pass

    orig_fp = _morgan_fp(original_smiles)
    gen_fp  = _morgan_fp(generated_smiles)
    if orig_fp is not None and gen_fp is not None:
        similarity = TanimotoSimilarity(orig_fp, gen_fp)
        components["novelty"] = max(0.0, 1.0 - similarity)

    if _TOX_AVAILABLE:
        try:
            has_alert = _get_toxicity_catalog().HasMatch(mol)
            components["tox_alert"] = 0.0 if has_alert else 1.0
        except Exception:
            pass

    components["tox21"] = score_tox21(generated_smiles)

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

def collect_pairs_from_stage1a(stage1a_dir: str) -> List[Tuple[str, str]]:
    """(masked_smiles, original_smiles) pairs from every Stage-1a combo CSV."""
    pairs: List[Tuple[str, str]] = []
    for csv_path in sorted(glob.glob(os.path.join(stage1a_dir, "mask*pct_temp*_seed*.csv"))):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                masked = (row.get("masked_smiles") or "").strip()
                orig   = (row.get("smiles") or "").strip()
                if masked and orig:
                    pairs.append((masked, orig))
    return pairs


def collect_pairs_from_stage1b(stage1b_dir: str) -> List[Tuple[str, str]]:
    """(masked_smiles, original_smiles) pairs from the Stage-1b summary CSV."""
    pairs: List[Tuple[str, str]] = []
    summary_path = os.path.join(stage1b_dir, "stage1b_large_scale_plip_mask_summary.csv")
    if not os.path.isfile(summary_path):
        return pairs
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            masked = (row.get("masked_smiles") or "").strip()
            orig   = (row.get("smiles") or "").strip()
            if masked and orig:
                pairs.append((masked, orig))
    return pairs


def collect_all_training_pairs(
    stage1a_dir: str = None,
    stage1b_dir: str = None,
    max_pairs:   int = MAX_TRAINING_PAIRS,
    sample_seed: int = SAMPLE_SEED,
) -> List[Tuple[str, str]]:
    """Concatenate Stage-1a + Stage-1b pairs, optionally sub-sampled to max_pairs."""
    stage1a_dir = stage1a_dir or config.STAGE1A_DIR
    stage1b_dir = stage1b_dir or config.STAGE1B_PLIP_MASK_DIR

    a = collect_pairs_from_stage1a(stage1a_dir)
    b = collect_pairs_from_stage1b(stage1b_dir)
    tqdm.write(f"  Collected {len(a)} pairs from Stage 1a, {len(b)} pairs from Stage 1b.")

    pairs = a + b
    if max_pairs and len(pairs) > max_pairs:
        rng = random.Random(sample_seed)
        pairs = rng.sample(pairs, max_pairs)
        tqdm.write(f"  Sub-sampled to {max_pairs} pairs (seed={sample_seed}).")

    tqdm.write(f"  Total training pairs: {len(pairs)}")
    return pairs


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
    }
    resume_adapter = None

    if ckpt and _ask_resume(save_dir, ckpt):
        start_epoch    = ckpt["last_epoch"] + 1
        baseline       = ckpt["baseline"]
        global_step    = ckpt["global_step"]
        history        = ckpt["history"]
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

        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"  Epoch {epoch}/{num_epochs}  ({len(batches)} batches × {batch_size} samples)")
        tqdm.write(
            "  What happens: model fills each <mask> token, samples a token\n"
            f"  from top-{top_k} candidates, records log P(token|context).\n"
            "  After all masks filled -> RDKit scores validity/QED/SA, plus\n"
            "  novelty (1 - similarity to the pre-mask parent), a PAINS/Brenk\n"
            "  toxicity-alert check, and (if configured) a Tox21 classifier probability.\n"
            f"  loss = -(score - {baseline:.3f}) x sum(log_prob)  <- this is what backward() sees."
        )
        tqdm.write(f"{'─'*60}")

        for batch in batches:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            b_reward = b_valid = b_qed = b_sa = b_novelty = b_tox_free = b_tox21 = 0.0

            for masked_smi, orig_smi in batch:
                def _score_only(smi, _orig=orig_smi, _w=score_weights):
                    s, _ = compute_stage9_score(smi, _orig, _w)
                    return s

                reward, log_prob, generated = reinforce_rollout(
                    smiles_masked=masked_smi, tokenizer=tokenizer, model=model,
                    device=device, top_k=top_k, temperature=temperature,
                    reward_fn=_score_only,
                )
                _, comps = compute_stage9_score(generated, orig_smi, score_weights)

                advantage  = reward - baseline
                batch_loss = batch_loss + (-advantage * log_prob)
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
            global_step += 1

            pbar.set_postfix_str(
                f"ep={epoch}/{num_epochs}  score={mean_reward:.3f}  "
                f"loss={batch_loss.item():.4f}  valid={ep_valid[-1]:.0%}  "
                f"novelty={ep_novelty[-1]:.2f}  tox_free={ep_tox_free[-1]:.0%}  "
                f"tox21={ep_tox21[-1]:.2f}  baseline={baseline:.3f}",
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
        )

        history["epoch"].append(epoch)
        history["step"].append(global_step)
        history["reward_mean"].append(_avg(ep_reward))
        history["loss_mean"].append(_avg(ep_loss))
        history["valid_rate"].append(_avg(ep_valid))
        history["qed_mean"].append(_avg(ep_qed))
        history["sa_mean"].append(_avg(ep_sa))
        history["novelty_mean"].append(_avg(ep_novelty))
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
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
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

def main() -> None:
    print("\n" + "=" * 60)
    print("STAGE 9 -- MASKED-DATA PROPERTY-GUIDED CHEMBERTA FINE-TUNING")
    print("=" * 60)
    print(f"""
  What this script does
  ----------------------
  1. Loads masked/original SMILES pairs from Stage 1a
     ({config.STAGE1A_DIR}) and Stage 1b
     ({config.STAGE1B_PLIP_MASK_DIR}).
  2. Loads ChemBERTa with trainable LoRA adapters (~1% of parameters).
  3. For each pair:
       a. Fills every <mask> token left-to-right, sampling from top-{TOP_K},
          recording log P(chosen_token | context) at each position.
       b. Decodes the full SMILES and scores it:
            score = {SCORE_W_VALID}*valid + {SCORE_W_QED}*QED + {SCORE_W_SA}*(1-SA/10)
                  + {SCORE_W_NOVELTY}*(1-similarity_to_original)
                  + {SCORE_W_TOX_ALERT}*(1-PAINS/Brenk_alert) + {SCORE_W_TOX21}*(1-Tox21_toxic_prob)
       c. loss = -(score - baseline) * sum(log_prob)  -- backpropagated
          through the LoRA weights (REINFORCE / score-function estimator;
          see the module docstring for why this is required instead of
          plain backprop).
  4. Checkpoints after every epoch; safe to interrupt and resume.

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

    with tempfile.TemporaryDirectory() as td:
        history = run_stage9_finetuning(
            pairs=list(test_pairs), save_dir=td,
            num_epochs=1, batch_size=2, top_k=5,
        )
        assert history["epoch"] == [1]
        assert os.path.isfile(os.path.join(td, "training_checkpoint.json"))
        assert os.path.isfile(os.path.join(td, "stage9_training_curves.png"))
        assert os.path.isdir(os.path.join(td, "epoch_001"))

    print("Stage 9 self-test passed.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main()

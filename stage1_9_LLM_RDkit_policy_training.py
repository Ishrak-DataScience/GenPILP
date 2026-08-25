# -*- coding: utf-8 -*-
"""
stage1_9_LLM_RDkit_policy_training.py
======================================
REINFORCE-based policy fine-tuning of ChemBERTa for molecule generation.

Design
------
ChemBERTa fills <mask> tokens sequentially left-to-right (same as Stage 2).
At every mask position the model produces a probability distribution; we
sample one token and record:

    log_prob_i = log P(chosen_token_i | all previously filled tokens)

After all masks are filled the complete SMILES is decoded and scored:

    reward = w_valid · valid  +  w_qed · QED  +  w_sa · (1 − SA/10)

The REINFORCE gradient estimator with a moving-average baseline is then:

    L = −(reward − baseline) · Σ log_prob_i

Only LoRA adapter weights are updated (peft), keeping ChemBERTa frozen.

Pause / Resume
--------------
A checkpoint is saved to <save_dir>/training_checkpoint.json after every
epoch.  Re-running the script detects the checkpoint and asks whether to
resume from the last completed epoch or start fresh.

Training data
-------------
  Stage 1 (IA) + Stage 1.5 (random) + Stage 1.7 (ChEMBL random), concatenated.

Usage
-----
  python stage1_9_LLM_RDkit_policy_training.py
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Silence all RDKit C++ warnings and info logs
RDLogger.DisableLog("rdApp.*")

import config

# ── tqdm (graceful fallback if not installed) ───────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):           # type: ignore[misc]
        return iterable if iterable is not None else range(0)

# ── Optional: SA score ──────────────────────────────────────────────────────
try:
    from rdkit.Contrib.SA_Score import sascorer
    _SA_AVAILABLE = True
except ImportError:
    _SA_AVAILABLE = False
    warnings.warn(
        "rdkit.Contrib.SA_Score not found — SA term will be 0.  "
        "Install with: pip install rdkit",
        stacklevel=1,
    )

# ── Dependency check: upgrade torchao before importing peft ────────────────
import subprocess as _sp
import importlib.metadata as _meta


def _ver_tuple(v: str) -> tuple:
    """Convert '0.16.0' → (0, 16, 0) for reliable comparison without packaging."""
    parts = []
    for seg in v.split(".")[:3]:
        try:
            parts.append(int(seg))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _ensure_torchao(min_version: str = "0.16.0") -> None:
    """Upgrade torchao only when the installed version is below min_version."""
    try:
        installed = _meta.version("torchao")
        if _ver_tuple(installed) >= _ver_tuple(min_version):
            return          # already satisfied — nothing to do
    except _meta.PackageNotFoundError:
        pass                # not installed at all → fall through to pip install
    tqdm.write(f"  Upgrading torchao to >={min_version} (required by peft) …")
    _sp.run(
        [sys.executable, "-m", "pip", "install", f"torchao>={min_version}", "-q"],
        check=True,
    )
    tqdm.write("  torchao upgrade complete.")


_ensure_torchao("0.16.0")

# ── Optional: peft (LoRA) ───────────────────────────────────────────────────
# Suppress harmless "Failed to load _C_mxfp8 / _C_cutlass" messages that
# torchao emits when optional GPU-kernel .so files are missing or built for
# a different Python ABI (common in Colab after a torchao upgrade).
import io as _io
import contextlib as _ctx

with _ctx.redirect_stderr(_io.StringIO()):
    try:
        import torchao as _torchao_preload  # noqa: F401  (pre-import to swallow .so warnings)
    except Exception:
        pass

try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    warnings.warn(
        "peft not installed — run: pip install peft\n"
        "Falling back to full fine-tuning (higher GPU memory).",
        stacklevel=1,
    )


# ════════════════════════════════════════════════════════════════════════════
#  DEFAULT HYPER-PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

LORA_RANK        = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05
LORA_TARGET_MODS = ["query", "value"]

REWARD_W_VALID   = 0.50
REWARD_W_QED     = 0.30
REWARD_W_SA      = 0.20

BASELINE_DECAY   = 0.99
TEMPERATURE      = 1.2
TOP_K            = 20
BATCH_SIZE       = 16
LEARNING_RATE    = 5e-5
NUM_EPOCHS       = 10
GRAD_CLIP        = 1.0

# Checkpoint filename written to save_dir after every epoch
_CKPT_FILE = "training_checkpoint.json"


# ════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT  (pause / resume)
# ════════════════════════════════════════════════════════════════════════════

def _ckpt_path(save_dir: str) -> str:
    return os.path.join(save_dir, _CKPT_FILE)


def _save_checkpoint(
    save_dir:    str,
    epoch:       int,
    baseline:    float,
    history:     Dict[str, list],
    optimizer:   torch.optim.Optimizer,
    global_step: int,
) -> None:
    """
    Save resumable training state to <save_dir>/training_checkpoint.json
    (metadata) and <save_dir>/optimizer.pt (optimizer tensor state).
    The LoRA adapter weights are already written by model.save_pretrained()
    each epoch under <save_dir>/epoch_{epoch:03d}/.
    """
    meta = {
        "last_epoch":   epoch,
        "baseline":     baseline,
        "global_step":  global_step,
        "history":      history,
    }
    with open(_ckpt_path(save_dir), "w") as f:
        json.dump(meta, f, indent=2)
    torch.save(optimizer.state_dict(),
               os.path.join(save_dir, "optimizer.pt"))


def _load_checkpoint(save_dir: str) -> Optional[dict]:
    """Return checkpoint dict or None if no checkpoint exists."""
    p = _ckpt_path(save_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _ask_resume(save_dir: str, ckpt: dict) -> bool:
    """Ask the user whether to resume from the checkpoint."""
    last = ckpt["last_epoch"]
    print(f"\n  Checkpoint found in: {save_dir}")
    print(f"  Last completed epoch : {last}")
    print(f"  Global step          : {ckpt['global_step']}")
    while True:
        ans = input("  Resume from checkpoint? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please enter Y or N.")


# ════════════════════════════════════════════════════════════════════════════
#  REWARD FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def compute_reward(
    smiles: str,
    w_valid: float = REWARD_W_VALID,
    w_qed:   float = REWARD_W_QED,
    w_sa:    float = REWARD_W_SA,
) -> float:
    """
    Composite RDKit reward for one SMILES string.
    Returns 0.0 immediately for invalid SMILES (RDKit can't parse them).
    All three components are in [0, 1]; final reward is clamped to [0, 1].
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    try:
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = 0.0

    if _SA_AVAILABLE:
        try:
            raw_sa = sascorer.calculateScore(mol)
            sa_val = max(0.0, (10.0 - raw_sa) / 10.0)
        except Exception:
            sa_val = 0.0
    else:
        sa_val = 0.0

    reward = w_valid * 1.0 + w_qed * qed_val + w_sa * sa_val
    return float(min(max(reward, 0.0), 1.0))


# ════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_chemberta_for_policy(
    model_name:      str         = config.CHEMBERTA_MODEL,
    lora_rank:       int         = LORA_RANK,
    lora_alpha:      int         = LORA_ALPHA,
    lora_dropout:    float       = LORA_DROPOUT,
    lora_targets:    List[str]   = LORA_TARGET_MODS,
    lora_checkpoint: Optional[str] = None,
    is_trainable:    bool          = True,
) -> Tuple[AutoTokenizer, torch.nn.Module, str]:
    """
    Load ChemBERTa and wrap it with LoRA adapters.

    If lora_checkpoint points to an existing directory the saved adapter
    is loaded from there (used when resuming training).

    `is_trainable` is forwarded to PeftModel.from_pretrained and defaults to
    True, which is what makes RESUME work at all. peft's own default is False:
    it assumes a saved adapter is being loaded for inference and sets
    requires_grad=False on every LoRA weight. A resuming caller then builds its
    optimizer from `filter(lambda p: p.requires_grad, model.parameters())`,
    gets an EMPTY list, and dies with "optimizer got an empty parameter list"
    -- which is exactly what every resume of Stage 1.9 / Stage 9 / Stage 9.1
    did before this argument existed.

    Pass False only for a pure inference load. It costs nothing to leave True
    there anyway: the evaluation paths run under torch.no_grad(), so no graph
    is built whatever the flag says.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write(f"  Device : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    base_model = AutoModelForMaskedLM.from_pretrained(model_name)

    if _PEFT_AVAILABLE:
        if lora_checkpoint and os.path.isdir(lora_checkpoint):
            model = PeftModel.from_pretrained(
                base_model, lora_checkpoint, is_trainable=is_trainable,
            )
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            tqdm.write(f"  LoRA adapter loaded from : {lora_checkpoint} "
                       f"({n_train:,} trainable params)")
            if is_trainable and n_train == 0:
                raise RuntimeError(
                    f"Adapter at {lora_checkpoint} loaded with zero trainable "
                    f"parameters despite is_trainable=True -- training cannot "
                    f"proceed. Check the peft version / adapter contents."
                )
        else:
            lora_cfg = LoraConfig(
                task_type      = TaskType.FEATURE_EXTRACTION,
                r              = lora_rank,
                lora_alpha     = lora_alpha,
                lora_dropout   = lora_dropout,
                target_modules = lora_targets,
                bias           = "none",
            )
            model = get_peft_model(base_model, lora_cfg)
            trainable, total = model.get_nb_trainable_parameters()
            tqdm.write(
                f"  LoRA adapters added — trainable params: "
                f"{trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
            )
    else:
        model = base_model
        tqdm.write("  Warning: peft unavailable — full fine-tune mode.")

    model = model.to(device)
    model.train()
    return tokenizer, model, device


# ════════════════════════════════════════════════════════════════════════════
#  REINFORCE ROLLOUT
# ════════════════════════════════════════════════════════════════════════════

def reinforce_rollout(
    smiles_masked: str,
    tokenizer,
    model,
    device:      str,
    top_k:       int   = TOP_K,
    temperature: float = TEMPERATURE,
    reward_fn          = compute_reward,
) -> Tuple[float, torch.Tensor, str]:
    """
    One REINFORCE episode: fill all <mask> tokens, score with RDKit.

    Returns
    -------
    reward    : scalar float from reward_fn
    log_prob  : scalar Tensor with grad_fn (sum of log P over mask positions)
    generated : decoded SMILES (may be invalid — reward_fn handles that)
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
        return reward_fn(clean), torch.tensor(0.0, requires_grad=False), clean

    log_prob_sum = torch.tensor(0.0, device=device)

    for pos in mask_positions:
        out    = model(input_ids=ids.unsqueeze(0),
                       attention_mask=attn_mask.unsqueeze(0))
        logits = out.logits[0, pos] / max(temperature, 1e-8)

        log_probs_full = F.log_softmax(logits, dim=-1)
        probs_full     = log_probs_full.exp()
        topk           = torch.topk(probs_full, k=min(top_k, logits.shape[0]))
        top_p          = topk.values / topk.values.sum()

        chosen_local = torch.multinomial(top_p.detach(), num_samples=1).item()
        chosen_id    = topk.indices[chosen_local]

        log_prob_sum = log_prob_sum + log_probs_full[chosen_id]
        ids[pos]     = chosen_id.detach()

    generated = tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
    reward    = reward_fn(generated)
    return reward, log_prob_sum, generated


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_policy_training(
    masked_smiles_list: List[str],
    model_name:      str   = config.CHEMBERTA_MODEL,
    save_dir:        str   = getattr(config, "RDKIT_POLICY_LORA_DIR",
                                     os.path.join(
                                         os.path.dirname(config.STAGE8_DIR),
                                         "stage2_policy_lora")),
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
    reward_weights:  Tuple[float, float, float] = (
        REWARD_W_VALID, REWARD_W_QED, REWARD_W_SA),
) -> Dict[str, list]:
    """
    REINFORCE fine-tuning loop with progress bars, checkpointing, and
    pause/resume support.

    Checkpoint strategy
    -------------------
    After every epoch:
      <save_dir>/epoch_{N:03d}/   — LoRA adapter weights for that epoch
      <save_dir>/optimizer.pt     — optimizer state (overwritten each epoch)
      <save_dir>/training_checkpoint.json  — epoch, baseline, history

    To resume: re-run the script; it detects the checkpoint and asks.
    To start fresh: answer 'n' to the resume prompt (or delete the checkpoint).
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Pause / resume detection ──────────────────────────────────────────────
    ckpt            = _load_checkpoint(save_dir)
    start_epoch     = 1
    baseline        = 0.0
    global_step     = 0
    history: Dict[str, list] = {
        "epoch": [], "step": [], "reward_mean": [], "loss_mean": [], "valid_rate": []
    }
    resume_adapter  = None

    if ckpt and _ask_resume(save_dir, ckpt):
        start_epoch  = ckpt["last_epoch"] + 1
        baseline     = ckpt["baseline"]
        global_step  = ckpt["global_step"]
        history      = ckpt["history"]
        resume_adapter = os.path.join(save_dir, f"epoch_{ckpt['last_epoch']:03d}")
        tqdm.write(f"\n  Resuming from epoch {start_epoch} "
                   f"(baseline={baseline:.3f}, step={global_step})")
    else:
        tqdm.write("\n  Starting fresh training run.")

    if start_epoch > num_epochs:
        tqdm.write("  Training already complete (all epochs done).")
        return history

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer, model, device = load_chemberta_for_policy(
        model_name      = model_name,
        lora_rank       = lora_rank,
        lora_alpha      = lora_alpha,
        lora_dropout    = lora_dropout,
        lora_targets    = lora_targets,
        lora_checkpoint = resume_adapter,
    )

    w_valid, w_qed, w_sa = reward_weights
    reward_fn = lambda smi: compute_reward(smi, w_valid, w_qed, w_sa)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    # Restore optimizer state if resuming
    opt_path = os.path.join(save_dir, "optimizer.pt")
    if resume_adapter and os.path.isfile(opt_path):
        try:
            optimizer.load_state_dict(
                torch.load(opt_path, map_location=device)
            )
            tqdm.write("  Optimizer state restored.")
        except Exception as e:
            tqdm.write(f"  Could not restore optimizer state: {e}")

    rng = random.Random(42 + start_epoch)

    remaining_epochs = num_epochs - start_epoch + 1
    total_batches    = (
        remaining_epochs
        * ((len(masked_smiles_list) + batch_size - 1) // batch_size)
    )

    # ── Main progress bar (all batches across all remaining epochs) ───────────
    pbar = tqdm(
        total     = total_batches,
        desc      = "Policy training",
        unit      = "batch",
        dynamic_ncols = True,
        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
    )

    for epoch in range(start_epoch, num_epochs + 1):
        rng.shuffle(masked_smiles_list)
        batches = [
            masked_smiles_list[i: i + batch_size]
            for i in range(0, len(masked_smiles_list), batch_size)
        ]

        epoch_rewards, epoch_losses, epoch_valids = [], [], []

        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"  Epoch {epoch}/{num_epochs}  "
                   f"({len(batches)} batches × {batch_size} samples)")
        tqdm.write(
            f"  What happens: model fills each <mask> token, samples a token\n"
            f"  from top-{top_k} candidates, records log P(token|context).\n"
            f"  After all masks filled → RDKit scores the SMILES.\n"
            f"  REINFORCE loss = −(reward − {baseline:.3f}) × Σ log_prob.\n"
            f"  Gradient update moves model toward higher-reward SMILES."
        )
        tqdm.write(f"{'─'*60}")

        for batch in batches:
            optimizer.zero_grad()
            batch_loss   = torch.tensor(0.0, device=device)
            batch_reward = 0.0
            batch_valid  = 0

            for smi_masked in batch:
                reward, log_prob, generated = reinforce_rollout(
                    smiles_masked = smi_masked,
                    tokenizer     = tokenizer,
                    model         = model,
                    device        = device,
                    top_k         = top_k,
                    temperature   = temperature,
                    reward_fn     = reward_fn,
                )
                advantage  = reward - baseline
                batch_loss = batch_loss + (-advantage * log_prob)
                batch_reward += reward
                if Chem.MolFromSmiles(generated) is not None:
                    batch_valid += 1

            batch_loss = batch_loss / max(len(batch), 1)
            batch_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            mean_reward = batch_reward / max(len(batch), 1)
            valid_rate  = batch_valid  / max(len(batch), 1)
            baseline    = (baseline_decay * baseline
                           + (1 - baseline_decay) * mean_reward)

            epoch_rewards.append(mean_reward)
            epoch_losses.append(batch_loss.item())
            epoch_valids.append(valid_rate)
            global_step += 1

            pbar.set_postfix_str(
                f"ep={epoch}/{num_epochs}  "
                f"reward={mean_reward:.3f}  "
                f"loss={batch_loss.item():.4f}  "
                f"valid={valid_rate:.0%}  "
                f"baseline={baseline:.3f}",
                refresh=True,
            )
            pbar.update(1)

        # ── Epoch summary ─────────────────────────────────────────────────────
        ep_r = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        ep_l = sum(epoch_losses)  / max(len(epoch_losses),  1)
        ep_v = sum(epoch_valids)  / max(len(epoch_valids),  1)

        tqdm.write(
            f"\n  Epoch {epoch} summary ──  "
            f"reward={ep_r:.3f}  loss={ep_l:.4f}  valid={ep_v:.1%}  "
            f"baseline={baseline:.3f}"
        )

        history["epoch"].append(epoch)
        history["step"].append(global_step)
        history["reward_mean"].append(ep_r)
        history["loss_mean"].append(ep_l)
        history["valid_rate"].append(ep_v)

        # ── Save epoch checkpoint (LoRA + optimizer + metadata) ───────────────
        epoch_adapter_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
        if _PEFT_AVAILABLE:
            model.save_pretrained(epoch_adapter_dir)
        else:
            model.save_pretrained(epoch_adapter_dir)
            tokenizer.save_pretrained(epoch_adapter_dir)

        _save_checkpoint(
            save_dir    = save_dir,
            epoch       = epoch,
            baseline    = baseline,
            history     = history,
            optimizer   = optimizer,
            global_step = global_step,
        )
        tqdm.write(f"  Checkpoint saved → {epoch_adapter_dir}")

    pbar.close()

    # ── Copy last epoch adapter to save_dir root (final model) ───────────────
    final_adapter = os.path.join(save_dir, f"epoch_{num_epochs:03d}")
    if os.path.isdir(final_adapter):
        import shutil
        for fname in os.listdir(final_adapter):
            shutil.copy2(
                os.path.join(final_adapter, fname),
                os.path.join(save_dir, fname),
            )
    tqdm.write(f"\n  Final LoRA adapter saved to : {save_dir}")

    _plot_training_history(history, save_dir)
    return history


# ════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def generate_smiles_with_policy(
    smiles_masked: str,
    tokenizer,
    model,
    device:      str,
    top_k:       int   = TOP_K,
    temperature: float = TEMPERATURE,
    num_samples: int   = config.INCREMENTAL_NUM_SAMPLES,
    save_path:   Optional[str] = None,
) -> List[str]:
    """
    Generate SMILES with the policy-adapted model (inference mode).
    Returns a sorted list of unique RDKit-valid SMILES.
    """
    model.eval()
    max_len = getattr(tokenizer, "model_max_length", 512)
    if max_len is None or max_len > 1024:
        max_len = 512

    enc = tokenizer(
        smiles_masked, return_tensors="pt",
        truncation=True, max_length=max_len,
    ).to(device)

    base_ids  = enc["input_ids"][0]
    base_attn = enc["attention_mask"][0]
    mask_id   = tokenizer.mask_token_id

    mask_positions = (base_ids == mask_id).nonzero(as_tuple=True)[0].tolist()
    if not mask_positions:
        clean = smiles_masked.replace(" ", "")
        return [clean] if Chem.MolFromSmiles(clean) is not None else []

    valid_set: set = set()
    with torch.no_grad():
        for _ in range(num_samples):
            ids = base_ids.clone()
            for pos in mask_positions:
                out    = model(input_ids=ids.unsqueeze(0),
                               attention_mask=base_attn.unsqueeze(0))
                logits = out.logits[0, pos] / max(temperature, 1e-8)
                probs  = torch.softmax(logits, dim=-1)
                topk   = torch.topk(probs, k=min(top_k, logits.shape[0]))
                top_p  = topk.values / topk.values.sum()
                chosen = torch.multinomial(top_p, num_samples=1).item()
                ids[pos] = topk.indices[chosen]

            cand = tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
            if Chem.MolFromSmiles(cand) is not None:
                valid_set.add(cand)

    valid_list = sorted(valid_set)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(valid_list) + "\n")

    model.train()
    return valid_list


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _use_stored_masked_smiles() -> bool:
    return bool(getattr(config, "USE_STORED_MASKED_SMILES", True))


def _resolve_masked_smiles(
    meta: dict,
    smiles: str,
    indices: Iterable[int],
    tokenizer,
) -> Tuple[str, bool]:
    """
    Return (masked_smiles, used_stored).

    Uses JSON ``masked_smiles`` when enabled and *indices* equals the full
    ``masked_atom_indices`` stored in that JSON (S1a full-mask records).
    Partial / incremental masks always recompute via mask_atoms_in_smiles.
    """
    from stage1_mask_calculation import mask_atoms_in_smiles

    idx_list = sorted(set(int(i) for i in indices))
    stored_full = sorted(set(meta.get("masked_atom_indices") or []))
    cached = (meta.get("masked_smiles") or "").strip()

    if (
        _use_stored_masked_smiles()
        and cached
        and idx_list == stored_full
    ):
        return cached, True

    return mask_atoms_in_smiles(smiles, idx_list, tokenizer), False


def collect_masked_smiles_from_stage1(
    stage1_dir:  str,
    stage15_dir: str,
    tokenizer,
) -> List[str]:
    """
    Build a list of masked SMILES strings for REINFORCE training from
    Stage-1 (IA) and Stage-1.5 (random) JSON files.

    Uses stored ``masked_smiles`` from JSON only when mc == N (full mask);
    incremental prefixes recompute (not stored in JSON).
    """

    def _load_jsons(folder: str) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
            try:
                with open(p, encoding="utf-8") as f:
                    meta = json.load(f)
                lig = meta.get("ligand", {})
                key = (f"{lig.get('resname','?')}-"
                       f"{lig.get('chain','?')}-"
                       f"{lig.get('resseq','?')}")
                result[key] = meta
            except Exception:
                pass
        return result

    s1  = _load_jsons(stage1_dir)
    s15 = _load_jsons(stage15_dir)
    masked_list: List[str] = []
    n_stored, n_recomputed = 0, 0

    for key in sorted(set(s1) & set(s15)):
        smiles      = s1[key]["smiles"]
        ia_indices  = s1[key]["masked_atom_indices"]
        rnd_indices = s15[key]["masked_atom_indices"]
        N = min(len(ia_indices), len(rnd_indices))
        for mc in range(1, N + 1):
            try:
                ia_masked, ia_from_json = _resolve_masked_smiles(
                    s1[key], smiles, ia_indices[:mc], tokenizer,
                )
                rnd_masked, rnd_from_json = _resolve_masked_smiles(
                    s15[key], smiles, rnd_indices[:mc], tokenizer,
                )
                masked_list.append(ia_masked)
                masked_list.append(rnd_masked)
                n_stored     += int(ia_from_json) + int(rnd_from_json)
                n_recomputed += int(not ia_from_json) + int(not rnd_from_json)
            except Exception:
                pass

    tqdm.write(
        f"  Collected {len(masked_list)} masked SMILES "
        f"from {len(set(s1) & set(s15))} PDB ligand(s) (Stage 1 + 1.5). "
        f"[stored={n_stored}, recomputed={n_recomputed}]"
    )
    return masked_list


def collect_masked_smiles_from_chembl(
    chembl_mask_dir: str,
    tokenizer,
) -> List[str]:
    """
    Build training samples from Stage-1.7 ChEMBL JSONs (one per file, 10A).

    Uses stored ``masked_smiles`` when present (full-mask ChEMBL records).
    """
    masked_list: List[str] = []
    n_stored, n_recomputed = 0, 0
    paths = sorted(glob.glob(os.path.join(chembl_mask_dir, "*.json")))

    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                meta = json.load(f)
            indices = meta.get("masked_atom_indices") or []
            if not indices:
                continue
            masked, from_json = _resolve_masked_smiles(
                meta, meta["smiles"], indices, tokenizer,
            )
            masked_list.append(masked)
            if from_json:
                n_stored += 1
            else:
                n_recomputed += 1
        except Exception:
            pass

    tqdm.write(
        f"  Collected {len(masked_list)} masked SMILES "
        f"from {len(paths)} ChEMBL JSON(s) (Stage 1.7). "
        f"[stored={n_stored}, recomputed={n_recomputed}]"
    )
    return masked_list


def collect_all_training_masked_smiles(tokenizer) -> List[str]:
    """Concatenate PDB (Stage 1 + 1.5) and ChEMBL (Stage 1.7) training data."""
    pdb_masked = collect_masked_smiles_from_stage1(
        stage1_dir  = config.MASK_CALC_OUTDIR,
        stage15_dir = config.RANDOM_MASK_OUTDIR,
        tokenizer   = tokenizer,
    )
    chembl_dir = getattr(
        config, "CHEMBL_MASK_OUTDIR",
        os.path.join(os.path.dirname(config.MASK_CALC_OUTDIR), "ChEMBL_Mask_Calculation"),
    )
    chembl_masked = collect_masked_smiles_from_chembl(chembl_dir, tokenizer)

    total = pdb_masked + chembl_masked
    tqdm.write(
        f"  Total training samples: {len(total)} "
        f"(PDB={len(pdb_masked)}, ChEMBL={len(chembl_masked)})"
    )
    return total


def _plot_training_history(history: Dict[str, list], save_dir: str) -> None:
    if not history.get("epoch"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["epoch"], history["reward_mean"],
                 marker="o", color="#1f77b4")
    axes[0].set_title("Mean Reward per Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Reward")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(history["epoch"], history["loss_mean"],
                 marker="s", color="#d62728")
    axes[1].set_title("Mean REINFORCE Loss per Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(history["epoch"], history["valid_rate"],
                 marker="^", color="#2ca02c")
    axes[2].set_title("RDKit Validity Rate per Epoch")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Valid rate")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(save_dir, "policy_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"  Training curves saved : {out}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 60)
    print("STAGE 1.9 — LLM RDKit POLICY TRAINING  (REINFORCE)")
    print("=" * 60)
    print(f"""
  What this script does
  ─────────────────────
  1. Loads masked SMILES from Stage-1 (IA), Stage-1.5 (random), and
     Stage-1.7 (ChEMBL BRD4 random) JSON files.
  2. Loads ChemBERTa with trainable LoRA adapters (~1% of parameters).
  3. For each masked SMILES (training sample):
       a. Fills every <mask> token left-to-right, sampling from top-{TOP_K}.
       b. Records log P(chosen_token | context) at each mask position.
       c. Decodes the full SMILES and scores it with RDKit:
            reward = {REWARD_W_VALID}·valid  +  {REWARD_W_QED}·QED  +  {REWARD_W_SA}·SA
       d. Computes REINFORCE loss = −(reward − baseline) × Σ log_prob.
       e. Updates LoRA weights via backprop.
  4. After each epoch saves a checkpoint so training can be paused and
     resumed at any time.

  Hyper-parameters
  ────────────────
  LoRA rank      : {LORA_RANK}   (target: {LORA_TARGET_MODS})
  Temperature    : {TEMPERATURE}     Top-k      : {TOP_K}
  Batch size     : {BATCH_SIZE}     LR         : {LEARNING_RATE}
  Epochs         : {NUM_EPOCHS}     Grad clip  : {GRAD_CLIP}
  SA scorer      : {"available" if _SA_AVAILABLE else "NOT available"}
  LoRA (peft)    : {"available" if _PEFT_AVAILABLE else "NOT available — pip install peft"}
""")

    tokenizer = AutoTokenizer.from_pretrained(config.CHEMBERTA_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    masked_smiles = collect_all_training_masked_smiles(tokenizer)

    if not masked_smiles:
        print("  No masked SMILES found.  Run stage1, stage1_5, and/or stage1_7.")
        sys.exit(1)

    save_dir = getattr(
        config, "RDKIT_POLICY_LORA_DIR",
        os.path.join(os.path.dirname(config.STAGE8_DIR), "stage2_policy_lora"),
    )

    history = run_policy_training(
        masked_smiles_list = masked_smiles,
        save_dir           = save_dir,
    )

    print("\n" + "=" * 60)
    print("  Policy training complete.")
    if history["valid_rate"]:
        print(f"  Final validity rate : {history['valid_rate'][-1]:.1%}")
        print(f"  Final mean reward   : {history['reward_mean'][-1]:.3f}")
    print(f"  LoRA adapter        : {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

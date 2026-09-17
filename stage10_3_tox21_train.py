# -*- coding: utf-8 -*-
"""
stage10_3_tox21_train.py
========================
Trains the Tox21 classifier that Stage 10.3's extra loss term reads.

WHY THIS FILE EXISTS
--------------------
Stage 10 states, at the top of stage10_vanila_backpropagation_training.py:

    "Tox21 is deliberately absent -- no checkpoint exists yet."

That is the only reason it was absent. Every other piece was already built:
stage9_masked_property_finetune.score_tox21 / score_tox21_batch know how to
run the classifier, compute_property_components(need_tox21=True) knows how to
carry the result, and Stage 9.1 already established the execution pattern
(RDKit in worker processes, the classifier as one batched forward on the
parent). The missing item was a directory to point STAGE9_TOX21_MODEL_DIR at.
This file produces that directory.

WHAT IT PRODUCES
----------------
A HuggingFace directory loadable by

    AutoTokenizer.from_pretrained(dir)
    AutoModelForSequenceClassification.from_pretrained(dir)

with 12 sigmoid outputs in config.STAGE9_TOX21_ALL_TASKS order, which is
exactly what stage9._load_tox21_classifier expects. Alongside the weights it
writes tox21_training_report.json (per-task AUC, split sizes, label counts,
every hyper-parameter) and two figures, so the number Stage 10.3 is about to
optimise against can be judged before it is trusted.

THE FOUR THINGS THAT MAKE THIS NON-TRIVIAL
-------------------------------------------
1. THE LABELS ARE SPARSE, AND THE BLANKS ARE NOT ZEROS.
   Measured on Tox21/tox21.csv: 7,831 molecules x 12 assays = 93,972 cells, of
   which ~78,900 are labelled. A blank means "this compound was never run in
   this assay", not "inactive". Imputing 0 -- the obvious thing, and what a
   plain BCELoss over a nan_to_num'd tensor effectively does -- teaches the
   model that ~16% of the training signal is confidently negative when it is
   unknown. Every loss here is MASKED: an unlabelled cell contributes nothing
   to the loss and nothing to the AUC.

2. THE POSITIVES ARE RARE, AND UNEVENLY SO.
   Per-task positive rates run 2.9% (NR-PPAR-gamma) to 16.2% (SR-ARE). Left
   alone, the minimiser finds "predict inactive everywhere", which scores well
   on accuracy, gives ~0.5 AUC, and -- this is the part that matters for Stage
   10.3 -- makes score_tox21 return a near-constant. A constant term added to
   every candidate cannot change which candidate wins best-of-K, so the whole
   exercise would silently do nothing. pos_weight = n_neg/n_pos per task,
   computed on the TRAIN split only, is applied per task.

3. THE SPLIT MUST BE SCAFFOLD-BASED, NOT RANDOM.
   Tox21 contains large families of near-identical compounds. A random split
   puts siblings on both sides and reports an AUC that will not survive
   contact with Stage 10.3's generated molecules, which are novel by
   construction (the novelty term pays for it). Bemis-Murcko scaffold split is
   the standard honest answer and typically costs 0.05-0.10 AUC against a
   random split -- that gap is the measurement, not a regression.

4. TRUNCATION MUST MATCH INFERENCE.
   stage9.score_tox21 tokenises with max_length=256. Training at a different
   length means the classifier sees whole molecules and then scores truncated
   ones. STAGE10_3_TOX21_MAX_TOKENS defaults to 256 for that reason, and this
   file WARNS if it is changed away from what score_tox21 uses.

RESUMING, AND THE ONE THING IT CANNOT MAKE EXACT
-------------------------------------------------
A full checkpoint (weights, AdamW's moments, the LR scheduler, the RNG streams,
the history and the best-so-far marker) is written after every epoch, atomically
and rolling, so a dropped Colab runtime or a pre-empted SLURM job costs at most
one epoch. Re-running the same command continues; --fresh starts over.

Continuing with the SAME --epochs is exact: the scheduler's state is restored
and the run proceeds as though it had never stopped.

Continuing with MORE --epochs is not, and cannot be. The LR schedule here is
OneCycleLR, whose shape is a function of total_steps -- so "6 epochs" and "10
epochs" are different functions, not a prefix and its extension. By the end of
a 6-epoch cycle the LR has annealed to ~0; a 10-epoch cycle is only 60% through
its anneal at that point. Extending therefore REBUILDS the schedule for the new
length and fast-forwards it to the current step, which raises the LR back up --
a warm restart. That is a legitimate thing to do and a different training
procedure from having run 10 epochs from the start, so it is announced with
both LR values printed rather than done quietly. STAGE10_3_TOX21_EXTEND_LR
picks the other option: hold the final LR flat for the added epochs instead.

WHAT IS DELIBERATELY NOT HERE
------------------------------
No sklearn. roc_auc below is the rank-based formula with tie averaging, ~20
lines and exact; adding a dependency to requirements.txt for one metric is a
worse trade than writing it down. No Trainer either -- every other training
loop in this repo is explicit, and matching them is worth more than the lines
saved.

Usage
-----
  python stage10_3_tox21_train.py
  python stage10_3_tox21_train.py --test           pure-function self-test, no
                                                   network, no GPU, no data
  ... --epochs 8 --batch-size 64 --lr 3e-5
  ... --csv Tox21/tox21.csv --out ./outputs/tox21_clf/
  ... --split random        random split instead of scaffold (reports a
                            higher, less honest AUC -- see point 3)
  ... --limit 500           tiny subset, for checking the loop runs at all
  ... --fresh               ignore any checkpoint and retrain from the base
  ... --no-resume           same as --fresh (kept for symmetry with Stage 10)

Then point config at it:
  STAGE9_TOX21_MODEL_DIR = "<the --out directory>"
  STAGE10_4_W_TOX21      = 0.20

and Stage 10.4 (stage10_4_tox21_aware_training.py) will read it. This file is
Stage 10.3 -- it trains the classifier; Stage 10.4 trains the generator that
uses it as a loss term.
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
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

RDLogger.DisableLog("rdApp.*")

import config

try:
    from tqdm import tqdm
except ImportError:                                   # pragma: no cover
    from tqdm_compat import tqdm  # type: ignore[misc]


# ── knobs ───────────────────────────────────────────────────────────────────
# The task list is Stage 9's, NOT a new one. The head's output order is the
# contract between this file and score_tox21, and there must be exactly one
# place it is written down.
TASKS: List[str] = list(getattr(config, "STAGE9_TOX21_ALL_TASKS", []))

CSV_PATH     = getattr(config, "STAGE10_3_TOX21_CSV", "Tox21/tox21.csv")
OUT_DIR      = getattr(config, "STAGE10_3_TOX21_MODEL_OUT",
                       "./outputs/stage10_3_tox21_classifier/")
BASE_MODEL   = (getattr(config, "STAGE10_3_TOX21_BASE_MODEL", None)
                or getattr(config, "CHEMBERTA_MODEL",
                           "seyonec/ChemBERTa-zinc-base-v1"))
EPOCHS       = getattr(config, "STAGE10_3_TOX21_EPOCHS", 6)
BATCH_SIZE   = getattr(config, "STAGE10_3_TOX21_BATCH_SIZE", 32)
LR           = getattr(config, "STAGE10_3_TOX21_LR", 2e-5)
WEIGHT_DECAY = getattr(config, "STAGE10_3_TOX21_WEIGHT_DECAY", 0.01)
GRAD_CLIP    = getattr(config, "STAGE10_3_TOX21_GRAD_CLIP", 1.0)
MAX_TOKENS   = getattr(config, "STAGE10_3_TOX21_MAX_TOKENS", 256)
SPLIT_MODE   = getattr(config, "STAGE10_3_TOX21_SPLIT", "scaffold")
VAL_FRAC     = getattr(config, "STAGE10_3_TOX21_VAL_FRAC", 0.10)
TEST_FRAC    = getattr(config, "STAGE10_3_TOX21_TEST_FRAC", 0.10)
SEED         = getattr(config, "STAGE10_3_TOX21_SEED", 42)
USE_POS_W    = getattr(config, "STAGE10_3_TOX21_POS_WEIGHT", True)
POS_W_CAP    = getattr(config, "STAGE10_3_TOX21_POS_WEIGHT_CAP", 50.0)
AUTO_RESUME  = getattr(config, "STAGE10_3_TOX21_AUTO_RESUME", None)
KEEP_CKPT    = getattr(config, "STAGE10_3_TOX21_KEEP_CHECKPOINT", True)
# What to do with the LR when a finished run is extended to more epochs.
#   "restart"  rebuild OneCycleLR for the new length and fast-forward to the
#              current step -- the LR jumps back up, a warm restart
#   "hold"     keep the LR the previous cycle ended on, flat, for the added
#              epochs -- a plain low-LR continuation
EXTEND_LR    = getattr(config, "STAGE10_3_TOX21_EXTEND_LR", "restart")

# The rolling checkpoint's filename, written inside out_dir next to the model.
CKPT_NAME = "tox21_train_checkpoint.pt"

# What stage9.score_tox21 hard-codes. Checked, not assumed -- a mismatch here
# is a silent train/serve skew, the kind that shows up as "the classifier was
# fine in training and useless in Stage 10.3".
_INFERENCE_MAX_TOKENS = 256


# ════════════════════════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════════════════════════

def load_tox21_csv(path: str = None,
                   tasks: Sequence[str] = None,
                   ) -> Tuple[List[str], np.ndarray]:
    """
    Read tox21.csv into (canonical SMILES, labels[N, T] with NaN for blanks).

    Reads the task columns BY NAME in `tasks` order rather than by position,
    so a re-ordered or extended CSV cannot silently permute the head's outputs
    -- which would be undetectable downstream, since every output is a
    plausible-looking probability either way.

    SMILES are canonicalised because that is what Stage 10.3 will feed the
    classifier: compute_property_components measures RDKit-parsed molecules,
    and score_tox21 sees the decoded generation. Training on raw CSV strings
    and serving canonical ones is a smaller skew than the truncation one, but
    it is free to remove.

    Duplicates (same canonical SMILES) are MERGED, not dropped: the same
    compound can appear twice with different assays filled in, and keeping the
    first row would throw away real labels. Conflicting labels for the same
    (molecule, assay) resolve to the positive, matching the assay convention
    that one confirmed active outweighs a non-response.
    """
    import csv

    path  = path or CSV_PATH
    tasks = list(tasks or TASKS)
    if not tasks:
        raise ValueError(
            "config.STAGE9_TOX21_ALL_TASKS is empty -- it defines the head's "
            "output order and must list the 12 Tox21 assays.")
    if not os.path.isfile(path):
        # STAGE10_3_TOX21_CSV is relative by default ("Tox21/tox21.csv"), which
        # resolves against the CWD -- and under Jupyter the CWD is the
        # notebook's directory, not the repo's. Falling back to the directory
        # this script lives in makes `python .../stage10_3_tox21_train.py` and a
        # notebook cell in some other folder both find the data, without anyone
        # having to hard-code an absolute path per machine.
        beside = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.isfile(beside):
            path = beside
        else:
            raise FileNotFoundError(
                f"Tox21 CSV not found at {path!r} (cwd {os.getcwd()!r}) nor at "
                f"{beside!r}.\nSet config.STAGE10_3_TOX21_CSV to an absolute "
                f"path, or pass --csv.")

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows.")

    missing = [t for t in tasks if t not in rows[0]]
    if missing:
        raise ValueError(
            f"{path} is missing task column(s) {missing}. Its columns are "
            f"{sorted(rows[0])}. The task list comes from "
            f"config.STAGE9_TOX21_ALL_TASKS and must match the CSV.")
    smi_col = "smiles" if "smiles" in rows[0] else "SMILES"
    if smi_col not in rows[0]:
        raise ValueError(f"{path} has no 'smiles' column.")

    by_smiles: Dict[str, np.ndarray] = {}
    order: List[str] = []
    n_unparsed = 0
    for row in rows:
        mol = Chem.MolFromSmiles(row[smi_col] or "")
        if mol is None:
            n_unparsed += 1
            continue
        canon = Chem.MolToSmiles(mol)
        vec = np.full(len(tasks), np.nan, dtype=np.float32)
        for j, t in enumerate(tasks):
            raw = (row[t] or "").strip()
            if raw not in ("", "NA", "nan"):
                try:
                    vec[j] = float(raw)
                except ValueError:
                    pass
        if canon in by_smiles:
            # NaN-aware max: an existing label beats a blank, and a 1 beats a
            # 0. np.fmax ignores NaN on either side, which is exactly the
            # "union the labels, positives win" rule.
            by_smiles[canon] = np.fmax(by_smiles[canon], vec)
        else:
            by_smiles[canon] = vec
            order.append(canon)

    if n_unparsed:
        print(f"  {n_unparsed} row(s) dropped: RDKit could not parse the SMILES.")
    n_merged = len(rows) - n_unparsed - len(order)
    if n_merged > 0:
        print(f"  {n_merged} duplicate canonical SMILES merged "
              f"(labels unioned, positives winning conflicts).")

    labels = np.stack([by_smiles[s] for s in order])
    return order, labels


def murcko_scaffold(smiles: str) -> str:
    """
    Bemis-Murcko scaffold as SMILES. Returns "" when RDKit cannot produce one;
    acyclic molecules legitimately have an empty scaffold, and grouping them
    together is correct -- they are all genuinely "no ring system".
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles,
                                                   includeChirality=False)
    except Exception:
        return ""


def scaffold_split(smiles: Sequence[str], val_frac: float = None,
                   test_frac: float = None, seed: int = None,
                   ) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministic Bemis-Murcko scaffold split into (train, val, test) indices.

    Whole scaffold GROUPS move together, so no compound in test shares a core
    with one in train. Groups are filled largest-first: the big scaffold
    families land in train, and the tail of singletons -- which is where the
    structural diversity is -- lands in val and test. That is the pessimistic
    assignment, and the one that predicts performance on Stage 10.3's
    generated molecules rather than flattering it.

    Deterministic given `seed`: the group ORDER is by (size desc, scaffold
    string), so ties break the same way on every machine, and the seed only
    shuffles within equal-size groups. A split that moved between runs would
    make the reported AUC unreproducible.
    """
    val_frac  = VAL_FRAC  if val_frac  is None else val_frac
    test_frac = TEST_FRAC if test_frac is None else test_frac
    rng = random.Random(SEED if seed is None else seed)

    groups: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        groups.setdefault(murcko_scaffold(smi), []).append(i)

    keys = list(groups)
    rng.shuffle(keys)                       # only breaks ties within a size
    keys.sort(key=lambda k: (-len(groups[k]), k))

    n = len(smiles)
    n_test = int(math.floor(n * test_frac))
    n_val  = int(math.floor(n * val_frac))
    train: List[int] = []
    val:   List[int] = []
    test:  List[int] = []
    for k in keys:
        g = groups[k]
        if len(test) + len(g) <= n_test:
            test.extend(g)
        elif len(val) + len(g) <= n_val:
            val.extend(g)
        else:
            train.extend(g)
    return sorted(train), sorted(val), sorted(test)


def random_split(smiles: Sequence[str], val_frac: float = None,
                 test_frac: float = None, seed: int = None,
                 ) -> Tuple[List[int], List[int], List[int]]:
    """
    Plain random split. Provided for comparison ONLY -- the gap between this
    and scaffold_split is the honest estimate of how much of the AUC comes
    from memorising compound families rather than learning chemistry.
    """
    val_frac  = VAL_FRAC  if val_frac  is None else val_frac
    test_frac = TEST_FRAC if test_frac is None else test_frac
    idx = list(range(len(smiles)))
    random.Random(SEED if seed is None else seed).shuffle(idx)
    n_test = int(math.floor(len(idx) * test_frac))
    n_val  = int(math.floor(len(idx) * val_frac))
    return (sorted(idx[n_test + n_val:]), sorted(idx[n_test:n_test + n_val]),
            sorted(idx[:n_test]))


# ════════════════════════════════════════════════════════════════════════════
#  LOSS AND METRIC
# ════════════════════════════════════════════════════════════════════════════

def masked_bce(logits: torch.Tensor, labels: torch.Tensor,
               pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    BCE-with-logits over the LABELLED cells only.

    `labels` carries NaN wherever the assay was not run. Those cells are
    replaced by 0 before the BCE call -- purely so the arithmetic stays finite,
    since 0 * nan is nan and would poison the whole batch's gradient -- and
    are then zeroed by the mask, so they contribute neither loss nor gradient.

    Normalised by the number of OBSERVED cells rather than the tensor size, so
    batches with different amounts of missingness are weighted equally per
    observation instead of per molecule.
    """
    mask = torch.isfinite(labels).float()
    safe = torch.nan_to_num(labels, nan=0.0)
    per_cell = F.binary_cross_entropy_with_logits(
        logits, safe, weight=None, pos_weight=pos_weight, reduction="none")
    denom = mask.sum().clamp(min=1.0)
    return (per_cell * mask).sum() / denom


def roc_auc(y_true: Sequence[float], y_score: Sequence[float]) -> Optional[float]:
    """
    ROC-AUC by the rank formula, with ties given average ranks.

        AUC = (sum_of_positive_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    Returns None when one class is absent, which is a real case here: a
    scaffold-split fold can hold zero positives for a 2.9%-prevalence assay.
    None means "not measurable" and is excluded from the mean rather than
    counted as 0.5 -- averaging in a number nobody measured is how a model
    that failed on the rare tasks comes to look average.
    """
    pairs = [(s, t) for s, t in zip(y_score, y_true)
             if s == s and t == t]                       # drops NaN on either
    n_pos = sum(1 for _, t in pairs if t >= 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    pairs.sort(key=lambda p: p[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                        # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    pos_rank_sum = sum(r for r, (_, t) in zip(ranks, pairs) if t >= 0.5)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def per_task_auc(labels: np.ndarray, probs: np.ndarray,
                 tasks: Sequence[str] = None) -> Dict[str, Optional[float]]:
    """AUC for each task independently, over that task's labelled cells."""
    tasks = list(tasks or TASKS)
    return {t: roc_auc(labels[:, j], probs[:, j]) for j, t in enumerate(tasks)}


def mean_auc(aucs: Dict[str, Optional[float]]) -> float:
    """Mean over the MEASURABLE tasks; NaN when none of them are."""
    vals = [v for v in aucs.values() if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


# ════════════════════════════════════════════════════════════════════════════
#  RESUMABILITY
# ════════════════════════════════════════════════════════════════════════════

def ckpt_path(out_dir: str) -> str:
    """The rolling checkpoint, beside the model it will eventually become."""
    return os.path.join(out_dir, CKPT_NAME)


def train_fingerprint(**kw) -> Dict[str, object]:
    """
    The settings a resume is only valid across if they are UNCHANGED.

    Split membership, step count and schedule shape all derive from these, so a
    checkpoint written under different ones does not describe the run being
    started. `epochs` is deliberately absent -- extending a finished run is the
    main reason to resume at all, and it is handled explicitly by
    _rebuild_scheduler rather than by refusing.

    n_train is in here as well as the fractions and the seed: it catches a CSV
    that changed underneath the run, which the other fields cannot see.
    """
    return {
        "base_model": kw["base_model"], "tasks": list(TASKS),
        "split_mode": kw["split_mode"], "seed": int(kw["seed"]),
        "val_frac": float(VAL_FRAC), "test_frac": float(TEST_FRAC),
        "max_tokens": int(kw["max_tokens"]), "batch_size": int(kw["batch_size"]),
        "lr": float(kw["lr"]), "n_train": int(kw["n_train"]),
        "pos_weight": bool(USE_POS_W), "pos_weight_cap": float(POS_W_CAP),
    }


def fingerprint_conflicts(old: Dict[str, object],
                          new: Dict[str, object]) -> List[str]:
    """Which fingerprint fields disagree. Empty means the resume is valid."""
    return [k for k in new if old.get(k) != new[k]]


def save_train_state(out_dir: str, *, model, optimizer, scheduler,
                     epoch: int, history: Dict[str, list],
                     best_val: float, best_epoch: Optional[int],
                     fingerprint: Dict[str, object],
                     total_steps: int, epochs_planned: int) -> None:
    """
    Write the rolling checkpoint atomically.

    ATOMIC BECAUSE THE FAILURE THIS EXISTS FOR IS A KILLED PROCESS. A plain
    torch.save straight onto the live path is not a safe operation: SIGKILL
    lands mid-write often enough to matter over a long run, and what survives
    is a truncated file that torch.load rejects -- i.e. the checkpoint is
    destroyed by the very event it was written to survive. Writing to a
    temporary name in the same directory and os.replace-ing it means the path
    holds either the old checkpoint or the new one, never a partial one.

    `epoch` is the NEXT epoch to run, matching stage10_lineage's convention, so
    a checkpoint written after epoch 6 of 6 holds 7 and is recognisably
    complete.

    SIZE, STATED: this is a FULL fine-tune, so every one of the ~44M parameters
    is trainable and AdamW keeps two moments for each. The file is therefore
    ~530 MB (weights 176 + moments 352), against the ~93 MB of Stage 10's
    checkpoints, which only carry the unfrozen tensors. It is rolling, so that
    cost is constant rather than per-epoch, and STAGE10_3_TOX21_KEEP_CHECKPOINT
    = False deletes it once the run finishes.
    """
    state = {
        "format": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "history": history,
        "best_val": float(best_val),
        "best_epoch": best_epoch,
        "fingerprint": fingerprint,
        "total_steps": int(total_steps),
        "epochs_planned": int(epochs_planned),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        },
        "saved_at": time.time(),
    }
    path = ckpt_path(out_dir)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_train_state(out_dir: str) -> Optional[dict]:
    """The checkpoint, or None when there is none or it will not load."""
    path = ckpt_path(out_dir)
    if not os.path.isfile(path):
        return None
    try:
        # weights_only=False: this carries RNG tuples and a history dict, not
        # only tensors. The file is one this script wrote, in a directory the
        # user owns.
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                             # pragma: no cover
        print(f"  Checkpoint at {path} could not be read ({e}); "
              f"starting fresh.")
        return None


def restore_rng(state: Optional[dict]) -> None:
    """Put the three RNG streams back. Dropout is why this matters."""
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu()
                            if hasattr(state["torch"], "cpu")
                            else state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:                             # pragma: no cover
        print(f"  Could not restore RNG state ({e}); continuing with a fresh one.")


def ask_resume(out_dir: str, ckpt: dict) -> bool:
    """
    Ask whether to resume, and auto-resume whenever stdin is not a TTY.

    Same convention as stage10_1._ask_resume, for the same reasons: a Colab
    `!python` cell or a SLURM batch job has no terminal, so a prompt could only
    raise EOFError or block forever, and an unattended job that was restarted
    overwhelmingly wants to continue. Resuming is also the non-destructive
    answer -- "no" retrains from the base model and overwrites this.
    """
    print(f"\n  Checkpoint found in : {out_dir}")
    print(f"  Next epoch          : {ckpt['epoch']}"
          f"  (of {ckpt.get('epochs_planned', '?')} planned last time)")
    print(f"  Best val mean AUC   : {ckpt.get('best_val', float('nan')):.4f}"
          f"  (epoch {ckpt.get('best_epoch')})")
    if ckpt.get("saved_at"):
        print(f"  Saved               : "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ckpt['saved_at']))}")

    if AUTO_RESUME:
        print("  config.STAGE10_3_TOX21_AUTO_RESUME is on -- resuming "
              "automatically. Pass --fresh to retrain instead.")
        return True
    if not sys.stdin.isatty():
        print("  Non-interactive session -- resuming automatically. "
              "Pass --fresh to retrain instead.")
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


def safe_pct_start(total_steps: int, pct_start: float = 0.1) -> float:
    """
    A warmup fraction that OneCycleLR can actually build for `total_steps`.

    torch places the warmup/anneal boundary at `pct_start * total_steps - 1`
    and then divides by (end_step - start_step). When that boundary lands
    exactly on 0 -- which pct_start=0.1 does at total_steps=10, and 0.2 does at
    5 -- the first get_lr() divides by zero and the CONSTRUCTOR raises, before
    a single batch has run.

    That is not a hypothetical: total_steps is steps_per_epoch x epochs, so
    `--limit 160 --batch-size 32 --epochs 2` hits it exactly. Raising the
    fraction until the boundary is at least 1 costs nothing on a real run (at
    1,470 steps the guard never binds) and turns a crash on a small smoke run
    into a slightly longer warmup.
    """
    total = max(int(total_steps), 1)
    return max(float(pct_start), 2.0 / total) if total > 2 else 0.5


def build_scheduler(optimizer, lr: float, total_steps: int,
                    done_steps: int = 0):
    """
    A OneCycleLR for `total_steps`, advanced to `done_steps`.

    Every scheduler in this file is built here, including the fresh-run one, so
    the guard above and the warmup fraction are defined once rather than in
    three places that could drift apart -- and a resumed schedule is
    necessarily the same curve as the one it is resuming.

    Fast-forwarding by calling step() in a loop rather than passing
    last_epoch=: OneCycleLR's last_epoch constructor path requires an
    'initial_lr' key that only exists in param groups a scheduler has already
    been attached to, so on a freshly-restored optimizer it raises. The loop is
    pure arithmetic on a closed-form curve -- a few thousand iterations is
    microseconds, and it cannot get out of step with the real schedule the way
    a re-derived formula could.
    """
    total = max(int(total_steps), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total,
        pct_start=safe_pct_start(total))
    for _ in range(max(0, min(int(done_steps), total - 1))):
        sched.step()
    return sched


# torch 2.0 renamed _LRScheduler to LRScheduler and kept the old name as an
# alias; resolving it here rather than importing one of them means this file
# does not pin a torch minor version for a base class it barely uses.
_LRSchedulerBase = getattr(torch.optim.lr_scheduler, "LRScheduler",
                           getattr(torch.optim.lr_scheduler, "_LRScheduler"))


class HoldLR(_LRSchedulerBase):
    """
    Keeps the LR exactly where the previous cycle left it.

    STAGE10_3_TOX21_EXTEND_LR = "hold" for extending a finished run: the added
    epochs continue at the small final LR instead of jumping back onto a
    rebuilt one-cycle. Fewer surprises in the loss curve, less chance of moving
    far from a good minimum -- and correspondingly less chance of escaping a
    mediocre one, which is what "restart" is for.
    """

    def get_lr(self):                                  # noqa: D102
        return [group["lr"] for group in self.optimizer.param_groups]


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ════════════════════════════════════════════════════════════════════════════

def compute_pos_weight(labels: np.ndarray, cap: float = None) -> torch.Tensor:
    """
    n_neg/n_pos per task, from the TRAIN split only.

    Capped (default 50) because NR-PPAR-gamma's ratio is ~33 already, and a
    fold holding fewer positives can push a task's weight into the hundreds --
    at which point that one assay's gradient drowns the other eleven. A task
    with no positives at all gets weight 1.0: there is nothing to up-weight.
    """
    cap = POS_W_CAP if cap is None else cap
    w = np.ones(labels.shape[1], dtype=np.float32)
    for j in range(labels.shape[1]):
        col = labels[:, j]
        col = col[np.isfinite(col)]
        n_pos = float((col >= 0.5).sum())
        n_neg = float(len(col) - n_pos)
        if n_pos > 0:
            w[j] = min(n_neg / n_pos, cap)
    return torch.tensor(w)


@torch.no_grad()
def predict(model, tokenizer, smiles: Sequence[str], device: str,
            batch_size: int = 64, max_tokens: int = None) -> np.ndarray:
    """Sigmoid probabilities [N, T], in input order."""
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    model.eval()
    out = []
    for i in range(0, len(smiles), batch_size):
        enc = tokenizer(list(smiles[i:i + batch_size]), return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=max_tokens).to(device)
        out.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    model.train()
    return (np.concatenate(out) if out
            else np.zeros((0, len(TASKS)), dtype=np.float32))


def train_tox21_classifier(
    csv_path:   str = None,
    out_dir:    str = None,
    base_model: str = None,
    epochs:     int = None,
    batch_size: int = None,
    lr:       float = None,
    max_tokens: int = None,
    split_mode: str = None,
    seed:       int = None,
    limit:      int = None,
    fresh:     bool = False,
    auto_resume: Optional[bool] = None,
) -> Dict[str, object]:
    """
    Fine-tune `base_model` into a 12-output Tox21 classifier and save it where
    STAGE9_TOX21_MODEL_DIR can point.

    Model selection is on VALIDATION mean AUC; the reported number is on the
    untouched TEST split. Keeping those separate matters more than usual here:
    this checkpoint is about to become a training signal for Stage 10.3, so an
    AUC chosen on the same data it is reported on would propagate an optimistic
    estimate straight into the generator's objective.

    THE BEST EPOCH IS SAVED WHEN IT HAPPENS, not held in memory until the end.
    Two reasons, and the second is the one that matters for a resumable run:
    a killed process leaves the best model already on disk rather than losing
    it, and `best_state` never has to live in the checkpoint, which would have
    added another 176 MB to a file that is already ~530 MB. The final test pass
    then loads the model back from out_dir, which has the useful side effect of
    evaluating the exact artefact score_tox21 will later load, rather than an
    in-memory object that is merely supposed to be identical to it.

    `fresh` ignores any checkpoint. `auto_resume` overrides the prompt: True
    resumes silently, False retrains. Left None it asks, and auto-resumes when
    stdin is not a terminal.
    """
    csv_path   = csv_path   or CSV_PATH
    out_dir    = out_dir    or OUT_DIR
    base_model = base_model or BASE_MODEL
    epochs     = EPOCHS     if epochs     is None else epochs
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    lr         = LR         if lr         is None else lr
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    split_mode = (split_mode or SPLIT_MODE).lower()
    seed       = SEED if seed is None else seed

    if max_tokens != _INFERENCE_MAX_TOKENS:
        print(f"\n  WARNING: training truncates at {max_tokens} tokens but "
              f"stage9.score_tox21 truncates at {_INFERENCE_MAX_TOKENS}. The "
              f"classifier would see different molecules at training and at "
              f"scoring time. Set STAGE10_3_TOX21_MAX_TOKENS = "
              f"{_INFERENCE_MAX_TOKENS} unless you are also changing "
              f"score_tox21.\n")

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"  Loading {csv_path} ...")
    smiles, labels = load_tox21_csv(csv_path)
    if limit:
        smiles, labels = smiles[:limit], labels[:limit]
    print(f"  {len(smiles)} unique molecules x {len(TASKS)} assays  "
          f"({int(np.isfinite(labels).sum())} labelled cells, "
          f"{100 * np.isfinite(labels).mean():.1f}% dense)")

    splitter = scaffold_split if split_mode == "scaffold" else random_split
    tr, va, te = splitter(smiles, seed=seed)
    print(f"  {split_mode} split -> train {len(tr)}  val {len(va)}  "
          f"test {len(te)}")
    if split_mode == "scaffold":
        n_scaf = len({murcko_scaffold(s) for s in smiles})
        print(f"  {n_scaf} distinct Bemis-Murcko scaffolds; none spans two "
              f"splits.")

    y_tr, y_va, y_te = labels[tr], labels[va], labels[te]
    x_tr = [smiles[i] for i in tr]
    x_va = [smiles[i] for i in va]
    x_te = [smiles[i] for i in te]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(TASKS),
        problem_type="multi_label_classification",
        # Written into config.json, so the saved checkpoint documents its own
        # output order. score_tox21 indexes by POSITION via
        # STAGE9_TOX21_ALL_TASKS; this is the record that lets a human confirm
        # the two agree without re-reading this script.
        id2label={i: t for i, t in enumerate(TASKS)},
        label2id={t: i for i, t in enumerate(TASKS)},
    ).to(device)
    model.train()

    pos_w = compute_pos_weight(y_tr).to(device) if USE_POS_W else None
    if pos_w is not None:
        worst = sorted(zip(TASKS, pos_w.tolist()), key=lambda p: -p[1])[:3]
        print(f"  pos_weight (train split, capped at {POS_W_CAP:g}): "
              + ", ".join(f"{t}={w:.1f}" for t, w in worst) + ", ...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(len(x_tr) / batch_size))
    n_steps = steps_per_epoch * max(epochs, 1)

    history: Dict[str, list] = {"epoch": [], "train_loss": [], "val_mean_auc": []}
    best_val, best_epoch = -1.0, None
    start_epoch = 1
    resumed = False
    fp = train_fingerprint(base_model=base_model, split_mode=split_mode,
                           seed=seed, max_tokens=max_tokens,
                           batch_size=batch_size, lr=lr, n_train=len(x_tr))

    # ── resume ────────────────────────────────────────────────────────────
    ckpt = None if fresh else load_train_state(out_dir)
    if ckpt is not None:
        conflicts = fingerprint_conflicts(ckpt.get("fingerprint") or {}, fp)
        if conflicts:
            # Not an error and not a silent overwrite: the weights are still
            # there, they just describe a different experiment. Saying which
            # field moved is the difference between "why is this retraining"
            # and a two-second fix.
            print(f"\n  A checkpoint exists in {out_dir} but was written under "
                  f"different settings\n  ({', '.join(conflicts)}). The data "
                  f"split, the step count or the schedule would\n  not match, "
                  f"so it cannot be continued -- retraining from "
                  f"{base_model}.\n  Use a different --out to keep both.")
            ckpt = None
        elif not (ask_resume(out_dir, ckpt) if auto_resume is None
                  else bool(auto_resume)):
            print("  Retraining from the base model; the checkpoint will be "
                  "overwritten.")
            ckpt = None

    scheduler = None
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"])
        history     = ckpt.get("history") or history
        best_val    = float(ckpt.get("best_val", -1.0))
        best_epoch  = ckpt.get("best_epoch")
        restore_rng(ckpt.get("rng"))
        resumed = True
        done_steps = (start_epoch - 1) * steps_per_epoch
        old_total = int(ckpt.get("total_steps", n_steps))

        if old_total == n_steps and ckpt.get("scheduler") is not None:
            # Same schedule, same length: an exact continuation.
            scheduler = build_scheduler(optimizer, lr, n_steps)
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # The run is being EXTENDED. OneCycleLR's shape is a function of
            # total_steps, so the old and new schedules are different curves --
            # not a prefix and its continuation. Whichever branch runs, say what
            # the LR did, because an unexplained jump in the loss curve two
            # epochs later is otherwise unattributable.
            lr_before = optimizer.param_groups[0]["lr"]
            if str(EXTEND_LR).lower() == "hold":
                scheduler = HoldLR(optimizer)
                print(f"\n  Extending {ckpt.get('epochs_planned')} -> {epochs} "
                      f"epochs. STAGE10_3_TOX21_EXTEND_LR='hold', so the added "
                      f"epochs\n  continue at the LR the previous cycle ended "
                      f"on ({lr_before:.3g}), flat.")
            else:
                scheduler = build_scheduler(optimizer, lr, n_steps, done_steps)
                lr_after = optimizer.param_groups[0]["lr"]
                print(f"\n  Extending {ckpt.get('epochs_planned')} -> {epochs} "
                      f"epochs. The one-cycle schedule is rebuilt for the new\n"
                      f"  length and fast-forwarded to step {done_steps}, which "
                      f"RAISES the learning rate\n  from {lr_before:.3g} to "
                      f"{lr_after:.3g} -- a warm restart, not a continuation of "
                      f"the old\n  curve. This is a different training procedure "
                      f"from running {epochs} epochs from\n  scratch. Set "
                      f"STAGE10_3_TOX21_EXTEND_LR = 'hold' to keep the low LR "
                      f"instead.")

        print(f"\n  Resuming at epoch {start_epoch}/{epochs}  "
              f"(best val mean AUC so far {best_val:.4f} at epoch {best_epoch})")

    if scheduler is None:
        scheduler = build_scheduler(optimizer, lr, n_steps)

    if start_epoch > epochs:
        print(f"\n  Training already complete: the checkpoint holds epoch "
              f"{start_epoch} and only {epochs} were\n  requested. Raise "
              f"--epochs to continue, or pass --fresh to retrain.")
        # Still fall through to the evaluation and report below, so re-running
        # a finished job re-emits its figures rather than doing nothing.

    order = list(range(len(x_tr)))

    if start_epoch <= epochs:
        print(f"\n  Training on {device} -- epochs {start_epoch}..{epochs}, "
              f"batch {batch_size}, lr {lr:g}, {steps_per_epoch} steps/epoch\n")
    for ep in range(start_epoch, epochs + 1):
        random.Random(seed + ep).shuffle(order)
        running, n_batches = 0.0, 0
        bar = tqdm(range(0, len(order), batch_size),
                   desc=f"  epoch {ep}/{epochs}", leave=False)
        for start in bar:
            sl = order[start:start + batch_size]
            enc = tokenizer([x_tr[i] for i in sl], return_tensors="pt",
                            padding=True, truncation=True,
                            max_length=max_tokens).to(device)
            yb = torch.tensor(y_tr[sl], device=device)

            optimizer.zero_grad()
            loss = masked_bce(model(**enc).logits, yb, pos_weight=pos_w)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            running += float(loss.detach())
            n_batches += 1
            bar.set_postfix_str(f"loss={running / n_batches:.4f}")

        val_probs = predict(model, tokenizer, x_va, device,
                            max_tokens=max_tokens)
        val_aucs = per_task_auc(y_va, val_probs)
        val_mean = mean_auc(val_aucs)

        history["epoch"].append(ep)
        history["train_loss"].append(running / max(n_batches, 1))
        history["val_mean_auc"].append(val_mean)
        print(f"  epoch {ep}/{epochs}  "
              f"train_loss={history['train_loss'][-1]:.4f}  "
              f"val_mean_AUC={val_mean:.4f}")

        if val_mean == val_mean and val_mean > best_val:
            best_val, best_epoch = val_mean, ep
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"    new best -- model saved to {out_dir}")

        # After the save-best, so a checkpoint always agrees with what is on
        # disk about which epoch was best.
        save_train_state(out_dir, model=model, optimizer=optimizer,
                         scheduler=scheduler, epoch=ep + 1, history=history,
                         best_val=best_val, best_epoch=best_epoch,
                         fingerprint=fp, total_steps=n_steps,
                         epochs_planned=epochs)

    if best_epoch is None:
        # No epoch ever produced a measurable validation AUC -- possible on a
        # tiny --limit run where the val fold holds no positives for any assay.
        # Save the final weights rather than leaving out_dir without a model.
        print("\n  No epoch produced a measurable validation AUC; saving the "
              "final weights.")
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    else:
        # Load the BEST epoch back for the test pass. from_pretrained rather
        # than an in-memory copy on purpose: this evaluates the exact files
        # score_tox21 will load, so a saving bug shows up here instead of
        # silently degrading Stage 10.3 later.
        print(f"\n  Loading the best epoch back from {out_dir} "
              f"(epoch {best_epoch}, val mean AUC {best_val:.4f}).")
        model = AutoModelForSequenceClassification.from_pretrained(
            out_dir).to(device)

    test_probs = predict(model, tokenizer, x_te, device, max_tokens=max_tokens)
    test_aucs  = per_task_auc(y_te, test_probs)
    test_mean  = mean_auc(test_aucs)

    print(f"\n  TEST mean AUC ({split_mode} split): {test_mean:.4f}")
    for j, t in enumerate(TASKS):
        v = test_aucs[t]
        n_lab = int(np.isfinite(y_te[:, j]).sum())
        n_pos = int((y_te[:, j] >= 0.5).sum())
        print(f"    {t:<16} AUC "
              + (f"{v:.3f}" if v is not None else "  n/a")
              + f"   (n={n_lab}, positives={n_pos})")

    # The model is NOT saved here: the best epoch already wrote it, and
    # re-saving would overwrite the best weights with whatever the last epoch
    # happened to produce -- exactly the bug the save-on-improvement flow
    # exists to prevent.

    report = {
        "base_model": base_model, "tasks": TASKS, "split": split_mode,
        "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "max_tokens": max_tokens, "pos_weight": bool(USE_POS_W),
        "pos_weight_cap": POS_W_CAP,
        "n_molecules": len(smiles),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "resumed": resumed, "best_epoch": best_epoch,
        "epochs_completed": len(history.get("epoch") or []),
        "extend_lr_policy": EXTEND_LR if resumed else None,
        "val_mean_auc_best": best_val,
        "test_mean_auc": test_mean,
        "test_auc_per_task": test_aucs,
        "history": history,
        "inference_contract": {
            "loaded_by": "stage9_masked_property_finetune._load_tox21_classifier",
            "task_order_source": "config.STAGE9_TOX21_ALL_TASKS",
            "score_tox21_returns": "1 - aggregated sigmoid over selected tasks",
        },
    }
    with open(os.path.join(out_dir, "tox21_training_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    _plot_training(history, out_dir)
    _plot_task_auc(test_aucs, y_te, out_dir, split_mode)

    if not KEEP_CKPT:
        try:
            os.remove(ckpt_path(out_dir))
            print(f"  Rolling checkpoint removed "
                  f"(STAGE10_3_TOX21_KEEP_CHECKPOINT = False).")
        except OSError:
            pass
    else:
        print(f"  Resume checkpoint     : {ckpt_path(out_dir)}  "
              f"(~530 MB; re-run with more --epochs to continue)")

    print(f"\n  Checkpoint saved -> {out_dir}")
    print(f"  Point config at it:\n"
          f"      STAGE9_TOX21_MODEL_DIR = {os.path.abspath(out_dir)!r}")
    return report


# ════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ════════════════════════════════════════════════════════════════════════════

def _plot_training(history: Dict[str, list], out_dir: str) -> None:
    """
    Loss and validation AUC against epoch, on SEPARATE axes.

    Not a twin-y single plot: the loss is unbounded above and the AUC lives in
    [0.5, 1.0], so sharing an axis makes whichever series has the smaller
    range look flat.
    """
    if not history.get("epoch"):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(history["epoch"], history["train_loss"], marker="o", color="#1f77b4")
    a1.set_title("Masked BCE (train)", fontsize=10)
    a1.set_xlabel("Epoch")
    a1.grid(True, linestyle="--", alpha=0.4)

    a2.plot(history["epoch"], history["val_mean_auc"], marker="o", color="#2ca02c")
    a2.axhline(0.5, color="#999999", linestyle=":", linewidth=1)
    a2.set_title("Validation mean ROC-AUC (measurable tasks)", fontsize=10)
    a2.set_xlabel("Epoch")
    a2.set_ylim(0.4, 1.0)
    a2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Stage 10.3 prerequisite -- Tox21 classifier training",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "tox21_classifier_training.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved : {out}")


def _plot_task_auc(aucs: Dict[str, Optional[float]], y_test: np.ndarray,
                   out_dir: str, split_mode: str) -> None:
    """
    Per-task test AUC as a bar chart, annotated with each task's positive
    count.

    The counts are ON the figure because a 0.85 AUC measured on 11 positives
    and one measured on 150 are not the same claim, and the bar alone cannot
    tell them apart. Unmeasurable tasks (no positives in the fold) are drawn
    as a grey stub at 0.5 rather than omitted, so a missing assay stays
    visible instead of silently shortening the axis.
    """
    names = list(aucs)
    if not names:
        return
    vals  = [(aucs[t] if aucs[t] is not None else 0.5) for t in names]
    known = [aucs[t] is not None for t in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), vals,
           color=["#2ca02c" if k else "#bbbbbb" for k in known])
    ax.axhline(0.5, color="#c0392b", linestyle="--", linewidth=1,
               label="random (0.5)")
    measurable = [v for v, k in zip(vals, known) if k]
    if measurable:
        ax.axhline(float(np.mean(measurable)), color="#1f77b4",
                   linestyle="-", linewidth=1.5,
                   label=f"mean {np.mean(measurable):.3f}")
    for i in range(len(names)):
        n_pos = int((y_test[:, i] >= 0.5).sum())
        ax.text(i, vals[i] + 0.01, f"n+={n_pos}", ha="center", fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Tox21 per-assay test AUC ({split_mode} split)  "
                 f"-- grey = no positives in the fold, not measurable",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(out_dir, "tox21_per_task_auc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-task AUC saved    : {out}")


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST  --  pure functions only: no network, no GPU, no dataset
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    print("\nStage 10.3 Tox21 trainer self-test")
    print("=" * 62)

    # 1. The task order is the contract with score_tox21.
    assert len(TASKS) == 12, \
        f"expected 12 Tox21 assays, config lists {len(TASKS)}"
    print("  [1] config.STAGE9_TOX21_ALL_TASKS holds 12 assays        OK")

    # 2. Masked BCE must IGNORE the blanks, not treat them as negatives.
    logits = torch.tensor([[2.0, -2.0, 5.0]])
    both   = torch.tensor([[1.0, 0.0, 0.0]])
    masked = torch.tensor([[1.0, 0.0, float("nan")]])
    l_both, l_masked = masked_bce(logits, both), masked_bce(logits, masked)
    assert l_both > l_masked, (
        "a confidently-wrong prediction on an UNLABELLED cell must not "
        "increase the loss")
    l_two = masked_bce(logits[:, :2], both[:, :2])
    assert abs(float(l_masked) - float(l_two)) < 1e-6, (
        f"masked loss {float(l_masked)} != loss over the labelled cells alone "
        f"{float(l_two)}")
    print("  [2] masked BCE ignores NaN cells exactly                 OK")

    # 3. NaN must not leak into the gradient.
    lg = torch.tensor([[0.5, -0.5]], requires_grad=True)
    masked_bce(lg, torch.tensor([[float("nan"), 1.0]])).backward()
    assert torch.isfinite(lg.grad).all(), "a NaN label poisoned the gradient"
    assert float(lg.grad[0, 0]) == 0.0, "an unlabelled cell produced gradient"
    print("  [3] unlabelled cells produce exactly zero gradient       OK")

    # 4. AUC against hand-computable cases.
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert abs(roc_auc([0, 1], [0.5, 0.5]) - 0.5) < 1e-12, "ties must give 0.5"
    assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None, (
        "a single-class column is NOT measurable and must return None")
    assert abs(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-12
    print("  [4] roc_auc: perfect / inverted / tied / single-class    OK")

    # 5. A task nobody could measure must not be averaged in as 0.5.
    m = mean_auc({"a": 0.9, "b": None, "c": 0.7})
    assert abs(m - 0.8) < 1e-12, f"unmeasurable task leaked into the mean: {m}"
    print("  [5] mean_auc excludes unmeasurable tasks                 OK")

    # 6. pos_weight from sparse labels, blanks excluded from BOTH counts.
    lab = np.array([[1.0], [0.0], [0.0], [0.0], [np.nan]], dtype=np.float32)
    w = compute_pos_weight(lab)
    assert abs(float(w[0]) - 3.0) < 1e-6, (
        f"expected n_neg/n_pos = 3/1, got {float(w[0])} -- the NaN row must "
        f"count as neither")
    assert float(compute_pos_weight(
        np.array([[0.0], [0.0]], np.float32))[0]) == 1.0
    print("  [6] pos_weight = n_neg/n_pos over labelled cells only    OK")

    # 7. Scaffold split: groups stay whole, and the split is reproducible.
    smis = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC",      # benzene scaffold
            "c1ccncc1C", "c1ccncc1CC",                      # pyridine scaffold
            "CCCC", "CCCCC", "CCO", "CC(C)O", "CCCCCC"]     # acyclic -> ""
    tr, va, te = scaffold_split(smis, val_frac=0.2, test_frac=0.2, seed=1)
    assert sorted(tr + va + te) == list(range(len(smis))), "split lost molecules"
    assert not (set(tr) & set(va)) and not (set(va) & set(te)) \
        and not (set(tr) & set(te)), "splits overlap"
    where: Dict[str, set] = {}
    for name, part in (("tr", tr), ("va", va), ("te", te)):
        for i in part:
            where.setdefault(murcko_scaffold(smis[i]), set()).add(name)
    straddling = {s for s, w_ in where.items() if len(w_) > 1}
    assert not straddling, (
        f"scaffold(s) {straddling} span more than one split -- the whole "
        f"point of a scaffold split is that they cannot")
    assert scaffold_split(smis, 0.2, 0.2, seed=1) == (tr, va, te), \
        "scaffold_split is not deterministic"
    print("  [7] scaffold split: whole groups, disjoint, reproducible OK")

    # 8. predict()'s contract, without loading a model: shape and batching.
    class _FakeModel:
        def __call__(self, **kw):
            n = kw["input_ids"].shape[0]

            class _O:
                pass

            o = _O()
            o.logits = torch.zeros(n, len(TASKS))
            return o

        def eval(self):
            return self

        def train(self):
            return self

    class _FakeTok:
        def __call__(self, xs, **kw):
            class _E(dict):
                def to(self, _):
                    return self

            return _E(input_ids=torch.zeros(len(xs), 4, dtype=torch.long))

    p = predict(_FakeModel(), _FakeTok(), ["C"] * 7, "cpu", batch_size=3)
    assert p.shape == (7, len(TASKS)), f"predict returned {p.shape}"
    print("  [8] predict returns [N, 12] across sub-batches           OK")

    # 9. The fingerprint must reject a resume that would change the SPLIT or
    #    the step count, and accept one that only changes the epoch count.
    base = dict(base_model="m", split_mode="scaffold", seed=1, max_tokens=256,
                batch_size=32, lr=2e-5, n_train=100)
    fp_a = train_fingerprint(**base)
    assert fingerprint_conflicts(fp_a, train_fingerprint(**base)) == [], \
        "identical settings must resume"
    assert "epochs" not in fp_a, (
        "epochs must NOT be in the fingerprint -- extending a finished run is "
        "the main reason to resume")
    for field, value in (("seed", 2), ("batch_size", 64), ("n_train", 101),
                         ("split_mode", "random"), ("max_tokens", 128),
                         ("lr", 3e-5), ("base_model", "other")):
        changed = dict(base)
        changed[field] = value
        got = fingerprint_conflicts(fp_a, train_fingerprint(**changed))
        assert field in got, (
            f"changing {field} moves the split or the step count and must "
            f"block a resume; conflicts reported: {got}")
    print("  [9] fingerprint blocks incompatible resumes, allows epochs   OK")

    # 10. The scheduler fast-forward has to land on the same LR the schedule
    #     would have reached by stepping there normally. If it does not, an
    #     extended run silently trains at the wrong learning rate.
    p = torch.nn.Parameter(torch.zeros(1))
    o1 = torch.optim.AdamW([p], lr=1e-3)
    ref = build_scheduler(o1, 1e-3, 100)
    for _ in range(40):
        ref.step()
    lr_stepped = o1.param_groups[0]["lr"]

    o2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    build_scheduler(o2, 1e-3, 100, done_steps=40)
    lr_ff = o2.param_groups[0]["lr"]
    assert abs(lr_stepped - lr_ff) < 1e-12, (
        f"fast-forward landed on {lr_ff}, stepping there gives {lr_stepped}")

    # And the warm-restart claim itself: extending really does raise the LR.
    o3 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    build_scheduler(o3, 1e-3, 60, done_steps=59)      # end of a 60-step cycle
    lr_end_short = o3.param_groups[0]["lr"]
    o4 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    build_scheduler(o4, 1e-3, 100, done_steps=59)     # same point, longer cycle
    lr_extended = o4.param_groups[0]["lr"]
    assert lr_extended > lr_end_short, (
        f"extending must raise the LR ({lr_end_short:.3g} -> "
        f"{lr_extended:.3g}); if it did not, the warning printed on resume "
        f"would be wrong")
    print(f"  [10] LR fast-forward exact; extending warm-restarts "
          f"{lr_end_short:.1e}->{lr_extended:.1e}  OK")

    # 11. HoldLR must hold, and must not reset the LR on construction.
    o5 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    o5.param_groups[0]["lr"] = 4.2e-6
    h = HoldLR(o5)
    for _ in range(5):
        h.step()
    assert abs(o5.param_groups[0]["lr"] - 4.2e-6) < 1e-12, (
        f"HoldLR changed the LR to {o5.param_groups[0]['lr']}")
    print("  [11] HoldLR holds the previous cycle's final LR              OK")

    # 12. The checkpoint round-trips, and the write is atomic.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        m = torch.nn.Linear(3, len(TASKS))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        sch = build_scheduler(opt, 1e-3, 10)
        sch.step()
        hist = {"epoch": [1], "train_loss": [0.5], "val_mean_auc": [0.71]}
        save_train_state(td, model=m, optimizer=opt, scheduler=sch, epoch=2,
                         history=hist, best_val=0.71, best_epoch=1,
                         fingerprint=fp_a, total_steps=10, epochs_planned=5)
        assert not os.path.exists(ckpt_path(td) + ".tmp"), (
            "the temporary file must be renamed away, not left behind")
        back = load_train_state(td)
        assert back["epoch"] == 2 and back["best_epoch"] == 1
        assert abs(back["best_val"] - 0.71) < 1e-12
        assert back["history"]["val_mean_auc"] == [0.71]
        assert back["scheduler"] is not None and back["total_steps"] == 10
        assert fingerprint_conflicts(back["fingerprint"], fp_a) == []
        # The restored scheduler must resume the same curve. Note WHAT is
        # being asserted: load_state_dict restores the scheduler's step
        # counter but does not write an LR into the optimizer until the next
        # step() -- in the real resume path the LR at the moment of restore
        # comes from optimizer.load_state_dict, which carries param_groups.
        # So the property that matters is that the NEXT step lands on the same
        # value, not that the LR matches the instant after loading.
        opt2 = torch.optim.AdamW(torch.nn.Linear(3, len(TASKS)).parameters(),
                                 lr=1e-3)
        sch2 = build_scheduler(opt2, 1e-3, 10)
        sch2.load_state_dict(back["scheduler"])
        assert sch2.last_epoch == sch.last_epoch, (
            f"restored step counter {sch2.last_epoch} != {sch.last_epoch}")
        sch.step()
        sch2.step()
        assert abs(opt2.param_groups[0]["lr"]
                   - opt.param_groups[0]["lr"]) < 1e-12, (
            f"a restored scheduler must continue the same LR curve: next step "
            f"gives {opt2.param_groups[0]['lr']} vs {opt.param_groups[0]['lr']}")
        assert load_train_state(tempfile.gettempdir() + "/nope_no_such") is None
    print("  [12] checkpoint save -> load round-trips, write is atomic    OK")

    print("\nStage 10.3 Tox21 trainer self-test passed.")
    print("  (The training loop itself needs the CSV and a model download; "
          "run without --test.)")


def _parse_args(argv: list) -> dict:
    """--csv, --out, --epochs, --batch-size, --lr, --split, --seed, --limit,
    --base-model, --max-tokens, --fresh / --no-resume."""
    out: Dict[str, object] = {}
    if "--fresh" in argv or "--no-resume" in argv:
        out["fresh"] = True
    flags = {"--csv": ("csv_path", str), "--out": ("out_dir", str),
             "--epochs": ("epochs", int), "--batch-size": ("batch_size", int),
             "--lr": ("lr", float), "--split": ("split_mode", str),
             "--seed": ("seed", int), "--limit": ("limit", int),
             "--base-model": ("base_model", str),
             "--max-tokens": ("max_tokens", int)}
    for flag, (key, cast) in flags.items():
        if flag not in argv:
            continue
        idx = argv.index(flag)
        if idx + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        out[key] = cast(argv[idx + 1])
    return out


def main() -> None:
    print("\n" + "=" * 62)
    print("STAGE 10.3 PREREQUISITE -- TOX21 CLASSIFIER")
    print("=" * 62)
    print(f"""
  Trains the checkpoint Stage 10.3's tox21 loss term reads. Without it,
  stage9.score_tox21 returns 0.0 for every molecule (its fail-safe), the
  term becomes a constant, and best-of-K selection is unchanged -- the
  Stage 10.3 run would look like it worked and optimise nothing.

    data     : {CSV_PATH}
    base     : {BASE_MODEL}
    split    : {SPLIT_MODE}   (scaffold = honest, random = flattering)
    labels   : masked BCE -- a blank is "not assayed", never "inactive"
    balance  : pos_weight per task from the train split (cap {POS_W_CAP:g})
    output   : {OUT_DIR}
    resume   : rolling checkpoint every epoch; re-run to continue, --fresh
               to retrain. Extending past the planned epoch count rebuilds
               the one-cycle LR ({EXTEND_LR!r}) -- see the module docstring.
""")
    train_tox21_classifier(**_parse_args(sys.argv))


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main()

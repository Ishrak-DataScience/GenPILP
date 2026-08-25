# -*- coding: utf-8 -*-
"""
stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py
==================================================================
Stage 9.1 -- Stage 9's property-guided LoRA fine-tuning, re-executed.

SAME objective, SAME data, SAME estimator. Only the execution strategy
differs, so the adapter this produces is directly comparable to Stage 9's and
the two can be run against each other as a pure speed/parity check.

Everything scientific is imported from stage9_masked_property_finetune rather
than re-implemented -- the score weights, the six-term composition, the pair
collection and per-parent capping, the checkpoint format, the plots. If the
objective is ever changed there, it changes here too. Nothing about the
chemistry is redefined in this file.

What is different: the two speedups Stage 9 leaves on the table
----------------------------------------------------------------
7a. BATCHED GPU FORWARD (reinforce_rollout_batched below)
    Stage 9 rolls out one molecule per forward pass: `ids.unsqueeze(0)`, batch
    dimension 1, sixteen times per optimizer step, plus sixteen more for the KL
    reference. Its BATCH_SIZE is gradient ACCUMULATION, not batching -- raising
    it buys no throughput at all, only memory.

    Stage 9.1 pads the whole batch into one [B, L] tensor and takes ONE forward
    for the policy and ONE for the reference. Every masked position in the
    batch is then sampled in a single `torch.multinomial` over a [M, k]
    matrix, and per-molecule log-probabilities are recovered with an
    `index_add_` over the flat mask index -- no Python loop over positions, and
    no loop over molecules.

    Correctness rests on padding not leaking into real positions. It does not:
    HF converts `attention_mask` into an additive `finfo.min` bias, so padded
    keys get exactly zero softmax weight, and RoBERTa derives position ids from
    a left-to-right cumsum over non-padding tokens, so trailing padding cannot
    shift a real token's position. Measured on this checkpoint, batched and
    unbatched logits at real positions agree to ~6e-06 -- float
    non-associativity across kernel shapes, not leakage. Both facts are pinned
    by _run_self_test, along with the padding_side=='right' precondition,
    because the one arrangement that DOES corrupt real tokens is left padding
    with a pad id that disagrees with model.config.pad_token_id.

7b. PARALLEL RDKit SCORING (ScoringPool below)
    RDKit is the bottleneck once the GPU work is batched: ~9 ms per molecule
    against ~1 ms of amortised forward. The molecules in a batch are scored
    independently, so they go to a process pool.

    The Tox21 term does NOT go to the workers -- it is a torch classifier, and
    one copy per worker would multiply memory by the pool size and contend for
    the GPU. It is scored on this process as a single batched forward
    (stage9.score_tox21_batch) and merged with the workers' RDKit measurements
    through stage9.compose_stage9_score, so the weighted sum still exists in
    exactly one place.

Reproducibility note
--------------------
Stage 9.1 will NOT reproduce a Stage 9 run token-for-token even at the same
seed, and that is expected rather than a defect. Batching changes how the
global RNG stream is consumed: Stage 9 draws one `multinomial` per mask
position in a fixed order, Stage 9.1 draws once for every mask in the batch
together. Same distribution, different sample -- like dealing from an
identically shuffled deck in a different order.

The two are therefore compared distributionally (mean validity/score over many
molecules, within confidence intervals), never by diffing outputs. The one
exception is `top_k=1`, which makes sampling deterministic; _run_self_test uses
exactly that to assert Stage 9.1 reproduces Stage 9's rollout EXACTLY -- same
SMILES, same log-probability -- which is what pins the gather/scatter/index_add
bookkeeping end to end.

Configuration
-------------
  config.STAGE9_1_LORA_DIR          own output dir; never clobbers Stage 9's
  config.STAGE9_1_BATCHED_ROLLOUT   "auto" (on when CUDA present) | True | False
  config.STAGE9_1_SCORING_WORKERS   "auto" | N | 0/1 for serial

Every other knob -- score weights, mask percent, per-parent cap, epochs, KL
beta, eval limit -- is Stage 9's, read from the same config entries.

Usage
-----
  python "stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py"
  python "stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py" --test
  ... --limit none        uncapped eval pass (match an uncapped Stage 9a baseline)
  ... --workers 0         force serial scoring
  ... --no-batch          force Stage 9's per-molecule rollout

NOTE ON THE FILENAME: it contains spaces and leads with a digit, so this file
cannot be `import`ed as a module -- it is a script only. That is why the pool
worker lives in stage9_1_scoring_worker.py, which has an importable name and
so survives the "spawn" start method Windows and Colab use.
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

import config
import stage9_1_scoring_worker as worker
from hardware_autotune import get_profile
from stage9_masked_property_finetune import (
    BASELINE_DECAY,
    BATCH_SIZE,
    GRAD_CLIP,
    KL_BASE_DROPOUT_OFF,
    KL_BETA,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODS,
    NUM_EPOCHS,
    SCORE_W_NOVELTY,
    SCORE_W_QED,
    SCORE_W_SA,
    SCORE_W_TOX21,
    SCORE_W_TOX_ALERT,
    SCORE_W_VALID,
    TEMPERATURE,
    TOP_K,
    _TOX21_AVAILABLE,
    collect_all_training_pairs,
    compose_stage9_score,
    compute_stage9_score,
    disable_base_dropout,
    reinforce_rollout_oneshot,
    run_property_distribution_eval,
    score_tox21_batch,
)
from stage1_9_LLM_RDkit_policy_training import (
    _ask_resume,
    _load_checkpoint,
    _save_checkpoint,
    load_chemberta_for_policy,
)

try:
    from tqdm import tqdm
except ImportError:                                  # pragma: no cover
    def tqdm(iterable=None, **kwargs):               # type: ignore[misc]
        return iterable if iterable is not None else range(0)

SCORE_WEIGHTS = (SCORE_W_VALID, SCORE_W_QED, SCORE_W_SA,
                 SCORE_W_NOVELTY, SCORE_W_TOX_ALERT, SCORE_W_TOX21)

MAX_MODEL_TOKENS = 512


# ════════════════════════════════════════════════════════════════════════════
#  RESOLVING THE TWO KNOBS
# ════════════════════════════════════════════════════════════════════════════

def resolve_batched(setting=None) -> bool:
    """
    config.STAGE9_1_BATCHED_ROLLOUT -> bool. "auto" means "on when CUDA is
    present": batching mainly recovers per-launch overhead, which dominates a
    batch-1 forward on GPU and matters much less on CPU.
    """
    if setting is None:
        setting = getattr(config, "STAGE9_1_BATCHED_ROLLOUT", "auto")
    if isinstance(setting, str) and setting.lower() == "auto":
        return torch.cuda.is_available()
    return bool(setting)


def resolve_workers(setting=None) -> int:
    """
    config.STAGE9_1_SCORING_WORKERS -> worker count (0 = serial, no pool).

    "auto" leaves one core for this process (which still runs the model, the
    tokenizer and the Tox21 batch) and caps at 8, past which RDKit scoring
    stops being the limiting factor. Returns 0 rather than 1 for a single
    worker: a one-worker pool is strictly slower than doing the work inline,
    since it adds pickling and IPC for no parallelism. This is the common case
    on a 2-vCPU Colab box -- measure there before trusting "auto".
    """
    if setting is None:
        setting = getattr(config, "STAGE9_1_SCORING_WORKERS", "auto")
    if isinstance(setting, str) and setting.lower() == "auto":
        # Defer to hardware_autotune, which reads the SLURM allocation / CPU
        # affinity / cgroup quota rather than os.cpu_count() -- on a shared HPC
        # node the host count is not what this job may use.
        setting = get_profile().cpu_workers
    n = int(setting)
    return 0 if n <= 1 else n


# ════════════════════════════════════════════════════════════════════════════
#  MULTI-GPU (DistributedDataParallel)
# ════════════════════════════════════════════════════════════════════════════
#
# THE TRAP, STATED UP FRONT: DDP does not make a step-limited run faster.
#
# DDP averages gradients across ranks, so N ranks each holding a batch of B
# behave as ONE optimizer step on an effective batch of N*B. Run 8 GPUs with
# the same per-rank batch you used on one and you get 8x fewer optimizer steps
# over the same data -- the run finishes sooner having learned less, which is
# the same trap as raising the batch size on a single card, only harder to see.
#
# So config.STAGE9_1_BATCH_SIZE is treated as the GLOBAL batch and split across
# ranks: per_rank = global // world_size. Step count is then IDENTICAL to a
# single-GPU run and the speedup is real wall-clock. _shard_batches enforces it.

def ddp_setup() -> Tuple[int, int, int, bool]:
    """
    Join the torchrun process group if we were launched under one.

    Returns (rank, world_size, local_rank, is_distributed). Falls back to
    single-process cleanly when RANK/WORLD_SIZE are absent, so the same script
    runs unchanged under `python ...` and `torchrun --nproc_per_node=8 ...`.

    nccl for GPU (the only backend with fast GPU all-reduce), gloo for CPU --
    which is what makes the DDP path testable on a machine with no GPUs at all.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or "RANK" not in os.environ:
        # Single-process: the correct path for 1 GPU, for CPU, and for a plain
        # `python ...` launch. Nothing below is needed and no process group is
        # created -- so the same script is valid unlaunched, under
        # `--nproc_per_node=1`, and under `--nproc_per_node=N`.
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpu > 1:
            # Easy to do by accident on an HPC node, and it silently wastes
            # 7/8 of the allocation, so say it rather than let it pass.
            warnings.warn(
                f"{n_gpu} GPUs are visible but this is a single process, so only "
                f"GPU 0 will be used. Launch with "
                f"`torchrun --nproc_per_node={n_gpu} <this script>` to use them all.",
                RuntimeWarning, stacklevel=2,
            )
        return 0, 1, 0, False

    import torch.distributed as dist
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_ws   = int(os.environ.get("LOCAL_WORLD_SIZE",
                                    os.environ.get("WORLD_SIZE", "1")))
    n_gpu      = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # More local ranks than GPUs is not something to paper over. NCCL requires
    # one rank per device; two ranks sharing cuda:0 typically HANGS in the
    # first all-reduce rather than failing, which is a miserable thing to
    # debug on a job queue. Fail immediately with the two ways out instead.
    if n_gpu and local_ws > n_gpu:
        raise RuntimeError(
            f"{local_ws} local ranks requested but only {n_gpu} GPU(s) visible. "
            f"NCCL needs one rank per GPU. Either use --nproc_per_node={n_gpu}, "
            f"or exercise the DDP logic on CPU with "
            f"CUDA_VISIBLE_DEVICES='' torchrun --nproc_per_node={local_ws} ... "
            f"(gloo backend, which is how to test this on a 1-GPU box like Colab)."
        )

    backend = "nccl" if n_gpu else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if n_gpu:
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank, True


def ddp_cleanup(is_dist: bool) -> None:
    if is_dist:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


def _all_reduce_mean(value: float, is_dist: bool, device: str) -> float:
    """Mean of `value` across ranks -- used to keep the REINFORCE baseline shared."""
    if not is_dist:
        return value
    import torch.distributed as dist
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / dist.get_world_size()


def _shard_batches(batches: List[list], rank: int, world_size: int,
                   is_dist: bool, device: str) -> List[list]:
    """
    Give each rank a disjoint slice, TRUNCATED so every rank runs the same
    number of steps.

    The truncation is not tidiness, it is a deadlock fix: DDP all-reduces on
    every backward, so a rank that runs out of batches early stops
    participating and every other rank blocks forever waiting for it. Ranks
    must agree on the step count before the loop starts, hence the MIN
    all-reduce. Length-bucketed batches vary in size, so this genuinely differs
    between ranks.
    """
    if not is_dist:
        return batches
    import torch.distributed as dist
    mine = batches[rank::world_size]
    n = torch.tensor([len(mine)], dtype=torch.long, device=device)
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    return mine[:int(n.item())]


def enable_tf32() -> bool:
    """TF32 matmuls on Ampere+ (no-op on T4/Turing and CPU). Free throughput."""
    if not getattr(config, "STAGE9_1_TF32", True) or not torch.cuda.is_available():
        return False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    except Exception:                                # pragma: no cover
        return False


def resolve_amp(device: str, setting=None):
    """
    config.STAGE9_1_AMP -> a torch dtype for autocast, or None for full fp32.

    "auto" prefers bf16 where the hardware supports it (Ampere and later: A100,
    L4, ...) because it needs no gradient scaler, and falls back to fp16 on
    Turing -- which is what a standard Colab GPU runtime gives you (T4). Always
    None on CPU, where autocast buys nothing here and would only complicate the
    equivalence self-test.
    """
    if setting is None:
        setting = getattr(config, "STAGE9_1_AMP", "auto")
    if not setting or _device_type(device) != "cuda":
        return None
    if isinstance(setting, str):
        s = setting.lower()
        if s == "bf16":
            return torch.bfloat16
        if s == "fp16":
            return torch.float16
        if s == "auto":
            # bf16 on Ampere+/Hopper (no scaler needed), fp16 on Volta/Turing.
            return get_profile().amp_dtype
    return None


def _token_lengths(masked_list: List[str], tokenizer) -> List[int]:
    """Padded-sequence length each masked SMILES would occupy, incl. specials."""
    enc = tokenizer(masked_list, add_special_tokens=True)
    return [len(x) for x in enc["input_ids"]]


def build_batches(
    pairs:      List[Tuple[str, str]],
    tokenizer,
    batch_size,
    rng:        random.Random,
    bucketing:  bool = None,
    max_tokens: int  = None,
    max_mols:   int  = None,
) -> List[List[Tuple[str, str]]]:
    """
    Split an epoch's pairs into batches, optionally length-bucketed.

    Two modes:
      batch_size = int      fixed molecules per batch (default 64)
      batch_size = "auto"   pack until molecules x longest-in-batch exceeds
                            max_tokens, so PADDED size -- and therefore GPU
                            memory -- stays roughly constant whether the batch
                            holds 8-token fragments or 500-token peptides

    Length bucketing sorts by token count first. That matters a lot here: the
    pairs run from ~8 to ~500 tokens, and one long SMILES in an unsorted batch
    pads every other row out to its length, so most of the [B, L, V] matrix is
    wasted compute.

    The BATCH ORDER is then shuffled, which is what keeps training stochastic
    despite the sort -- otherwise every epoch would walk from shortest molecule
    to longest in the same order, and the gradient sequence would be a length
    curriculum rather than a random one. Within-batch length correlation
    remains, which is the accepted cost of bucketing.
    """
    if bucketing is None:
        bucketing = getattr(config, "STAGE9_1_LENGTH_BUCKETING", True)
    if max_tokens is None:
        max_tokens = getattr(config, "STAGE9_1_MAX_BATCH_TOKENS", 65536)
    if max_mols is None:
        max_mols = getattr(config, "STAGE9_1_MAX_BATCH_MOLECULES", 512)

    idx = list(range(len(pairs)))
    auto = isinstance(batch_size, str) and batch_size.lower() == "auto"

    if bucketing or auto:
        lengths = _token_lengths([m for m, _ in pairs], tokenizer)
        rng.shuffle(idx)                       # break ties randomly, not by file order
        idx.sort(key=lambda i: lengths[i])
    else:
        rng.shuffle(idx)

    batches: List[List[int]] = []
    if auto:
        cur: List[int] = []
        cur_max = 0
        for i in idx:
            new_max = max(cur_max, lengths[i])
            if cur and ((len(cur) + 1) * new_max > max_tokens
                        or len(cur) + 1 > max_mols):
                batches.append(cur)
                cur, cur_max, new_max = [], 0, lengths[i]
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
    else:
        b = int(batch_size)
        batches = [idx[i:i + b] for i in range(0, len(idx), b)]

    rng.shuffle(batches)
    return [[pairs[i] for i in b] for b in batches]


# ════════════════════════════════════════════════════════════════════════════
#  7a -- BATCHED ROLLOUT
# ════════════════════════════════════════════════════════════════════════════

_NO_REFERENCE_WARNED = False


def _device_type(device: str) -> str:
    """"cuda:3" -> "cuda". autocast and GradScaler take a device TYPE, not an
    indexed device, and under DDP every rank past 0 holds an indexed one."""
    return str(device).split(":")[0]


def _autocast(device: str, amp_dtype):
    """autocast when a mixed-precision dtype is active, otherwise a no-op."""
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=_device_type(device), dtype=amp_dtype)


def _reference_logits_batched(model, ids: torch.Tensor, attn_mask: torch.Tensor,
                              amp_dtype=None):
    """
    Frozen-pretrained logits for a whole [B, L] batch -- the KL anchor, read
    from the same still-masked input as the policy pass.

    Identical in intent to Stage 9's _reference_logits (adapter switched off,
    eval() so the anchor is deterministic, no second checkpoint to drift), just
    without the per-molecule unsqueeze.
    """
    global _NO_REFERENCE_WARNED
    if not hasattr(model, "disable_adapter"):
        if not _NO_REFERENCE_WARNED:
            _NO_REFERENCE_WARNED = True
            warnings.warn(
                "KL beta > 0 but the model has no LoRA adapter to disable "
                "(peft unavailable?), so no frozen reference exists -- the KL "
                "anchor is inactive for this run.",
                RuntimeWarning,
            )
        return None

    was_training = model.training
    model.eval()
    try:
        device = "cuda" if ids.is_cuda else "cpu"
        with torch.no_grad(), model.disable_adapter(), _autocast(device, amp_dtype):
            return model(input_ids=ids, attention_mask=attn_mask).logits
    finally:
        if was_training:
            model.train()


def reinforce_rollout_batched(
    masked_list: List[str],
    tokenizer,
    model,
    device:      str,
    top_k:       int   = TOP_K,
    temperature: float = TEMPERATURE,
    kl_beta:     float = 0.0,
    amp_dtype          = None,
    ref_model          = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    One-shot REINFORCE rollout for a WHOLE BATCH in two forward passes.

    Stage 9's reinforce_rollout_oneshot does this per molecule: 2B forwards for
    B molecules. Here it is 2 forwards total, and the per-position Python loop
    is gone as well -- every masked position in the batch is sampled in one
    `multinomial`.

    Returns
    -------
    log_probs : [B] Tensor with grad_fn -- per molecule, the SUM of
                log P(chosen token | context) over that molecule's masks.
                Recovered from the flat per-mask vector with index_add_ over
                the batch index, which is what keeps molecules from bleeding
                into each other's credit.
    kls       : [B] Tensor with grad_fn -- per molecule, the summed exact KL
                against the frozen model over its masked positions. All zeros
                (and no reference pass) when kl_beta == 0. Computed on RAW
                logits, deliberately not the temperature-scaled/top-k ones:
                the anchor constrains the MODEL, not the sampling policy laid
                over it, matching Stage 9 exactly.
    generated : list of B decoded SMILES (any of which may be invalid --
                scoring handles that).

    Molecules with no <mask> at all get log_prob 0 and kl 0 and decode to
    themselves, the same no-op Stage 9 returns for that case.
    """
    if tokenizer.padding_side != "right":
        # Left padding shifts RoBERTa's position ids whenever the pad id and
        # model.config.pad_token_id disagree, and silently corrupts real
        # tokens. Right padding is the HF default; refuse rather than trust it.
        raise ValueError(
            f"tokenizer.padding_side must be 'right' for batched rollout, got "
            f"{tokenizer.padding_side!r}."
        )

    max_len = getattr(tokenizer, "model_max_length", MAX_MODEL_TOKENS)
    if max_len is None or max_len > 1024:
        max_len = MAX_MODEL_TOKENS

    enc = tokenizer(
        masked_list, return_tensors="pt", padding=True,
        truncation=True, max_length=max_len,
    ).to(device)
    ids       = enc["input_ids"]                       # [B, L]
    attn_mask = enc["attention_mask"]                  # [B, L]
    B         = ids.shape[0]

    with _autocast(device, amp_dtype):
        out = model(input_ids=ids, attention_mask=attn_mask)
    # The KL anchor reads from the UNWRAPPED model under DDP: disable_adapter
    # lives on the PeftModel, and a second forward through the DDP wrapper
    # would register autograd hooks for a pass we never back-propagate.
    ref_logits = (_reference_logits_batched(ref_model or model, ids, attn_mask,
                                            amp_dtype)
                  if kl_beta > 0.0 else None)

    log_probs = torch.zeros(B, device=device)
    kls       = torch.zeros(B, device=device)

    # Flat index of every masked position across the batch. Padding is never
    # selected here: pad tokens are not mask tokens.
    b_idx, l_idx = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)

    filled = ids.clone()
    if b_idx.numel() > 0:
        # .float() before any softmax: under fp16 autocast a log_softmax over a
        # 767-wide vocabulary loses meaningful precision, and these values feed
        # both the gradient and the KL. The cast is negligible next to the
        # forward pass -- it touches only the masked rows, not [B, L, V].
        sel   = out.logits[b_idx, l_idx].float()               # [M, V]
        V     = sel.shape[-1]
        k     = min(top_k, V)

        logp_full = F.log_softmax(sel / max(temperature, 1e-8), dim=-1)  # [M, V]
        topk      = torch.topk(logp_full.exp(), k=k, dim=-1)             # [M, k]
        top_p     = topk.values / topk.values.sum(dim=-1, keepdim=True)

        # ONE draw for every mask in the batch. multinomial on a 2-D input
        # samples one index per row, so this replaces both the per-position
        # and the per-molecule loop at once.
        choice     = torch.multinomial(top_p.detach(), num_samples=1)     # [M, 1]
        chosen_ids = topk.indices.gather(1, choice).squeeze(1)            # [M]

        # log P of the token actually taken, read off the FULL-vocab
        # log-softmax (not the renormalised top-k one) exactly as Stage 9 does.
        chosen_logp = logp_full.gather(1, chosen_ids.unsqueeze(1)).squeeze(1)
        log_probs   = log_probs.index_add(0, b_idx, chosen_logp)

        filled[b_idx, l_idx] = chosen_ids.detach()

        if ref_logits is not None:
            lp_policy = F.log_softmax(out.logits[b_idx, l_idx].float(), dim=-1)
            lp_ref    = F.log_softmax(ref_logits[b_idx, l_idx].float(), dim=-1)
            kl_per_mask = (lp_policy.exp() * (lp_policy - lp_ref)).sum(dim=-1)
            kls = kls.index_add(0, b_idx, kl_per_mask)

    generated = [s.replace(" ", "")
                 for s in tokenizer.batch_decode(filled, skip_special_tokens=True)]
    return log_probs, kls, generated


# ════════════════════════════════════════════════════════════════════════════
#  7b -- PARALLEL RDKit SCORING
# ════════════════════════════════════════════════════════════════════════════

class ScoringPool:
    """
    A persistent process pool for RDKit scoring, or a transparent no-op when
    workers <= 0.

    Persistent on purpose: under "spawn" every worker re-imports the module
    tree (torch included), which costs seconds. Paying that once per RUN is
    fine; paying it once per BATCH would dwarf the work being parallelised.
    Use as a context manager so the pool is always torn down, including on an
    interrupted run.
    """

    def __init__(self, workers: int):
        self.workers = max(0, int(workers))
        self.pool = None

    def __enter__(self) -> "ScoringPool":
        if self.workers > 0:
            # Explicit spawn context so behaviour is identical on Linux and
            # Windows rather than silently forking on one and not the other.
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(
                processes=self.workers, initializer=worker.init_worker,
            )
        return self

    def __exit__(self, *exc) -> None:
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def measure(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Optional[float]]]:
        """RDKit measurements for (generated, parent) pairs, in input order."""
        if not pairs:
            return []
        if self.pool is None:
            return worker.score_many(pairs)
        # chunksize so each worker gets a contiguous slice rather than paying
        # per-item IPC on a batch this small.
        chunk = max(1, len(pairs) // (self.workers * 2) or 1)
        return self.pool.map(worker.score_one, pairs, chunksize=chunk)


def score_batch(
    generated: List[str],
    parents:   List[str],
    pool:      "ScoringPool",
    weights:   Tuple[float, ...] = SCORE_WEIGHTS,
) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Score a batch: RDKit terms in the pool, the Tox21 term here as one batched
    forward, merged through stage9.compose_stage9_score.

    Splitting it this way is what keeps the two halves honest -- the workers
    never load a torch model, and the weighted sum that defines the objective
    is still written down exactly once, in Stage 9.
    """
    measured = pool.measure(list(zip(generated, parents)))

    if _TOX21_AVAILABLE:
        for m, clean in zip(measured, score_tox21_batch(generated)):
            m["tox21"] = clean

    scored = [compose_stage9_score(m, weights) for m in measured]
    return [s for s, _ in scored], [c for _, c in scored]


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_stage9_1_finetuning(
    pairs:          List[Tuple[str, str]],
    model_name:     str   = config.CHEMBERTA_MODEL,
    save_dir:       str   = None,
    num_epochs:     int   = NUM_EPOCHS,
    batch_size            = None,
    lr:             float = LEARNING_RATE,
    temperature:    float = TEMPERATURE,
    top_k:          int   = TOP_K,
    grad_clip:      float = GRAD_CLIP,
    baseline_decay: float = BASELINE_DECAY,
    kl_beta:        float = KL_BETA,
    score_weights:  Tuple[float, ...] = SCORE_WEIGHTS,
    batched:        bool  = None,
    workers:        int   = None,
) -> Dict[str, list]:
    """
    Stage 9's REINFORCE loop with the batched rollout and the scoring pool.

    The loss is identical term for term:

        loss = mean_over_batch[ -(score - baseline) * sum(log_prob) ]
               + kl_beta * mean_over_batch[ KL(policy || pretrained) ]

    Checkpoint layout is Stage 9's, written by the same helpers, so a run can
    be inspected or resumed with the same tooling.
    """
    save_dir = save_dir or getattr(config, "STAGE9_1_LORA_DIR", None) \
        or config.STAGE9_LORA_DIR
    os.makedirs(save_dir, exist_ok=True)

    batched = resolve_batched() if batched is None else bool(batched)
    workers = resolve_workers() if workers is None else resolve_workers(workers)
    if batch_size is None:
        batch_size = getattr(config, "STAGE9_1_BATCH_SIZE", BATCH_SIZE)

    rank, world_size, local_rank, is_dist = ddp_setup()
    is_main = (rank == 0)

    def log(msg: str) -> None:
        """Only rank 0 prints, or 8 ranks interleave the same lines 8 times."""
        if is_main:
            tqdm.write(msg)

    hw = get_profile()
    hw.apply()                       # torch threads, TF32, matmul precision
    if isinstance(batch_size, str) and batch_size.lower() == "hardware":
        # Sized from the dataset and epoch budget, not from GPU memory -- see
        # HardwareProfile.recommended_batch_size for why memory is the wrong
        # target for a 44M-parameter model.
        batch_size = hw.recommended_batch_size(
            seq_len=128, vocab=767, n_pairs=len(pairs), n_epochs=num_epochs,
        )
        log(f"  Batch size {batch_size} chosen by hardware_autotune "
            f"(config.STAGE9_1_BATCH_SIZE='hardware').")

    # config batch size is the GLOBAL batch; split it so the optimizer-step
    # count matches a single-GPU run. See the DDP note above.
    global_batch = batch_size
    if is_dist and not isinstance(batch_size, str):
        batch_size = max(1, int(batch_size) // world_size)
        log(f"  DDP: global batch {global_batch} split across {world_size} "
            f"rank(s) -> {batch_size} per rank (step count unchanged).")

    # Divide the CPU allocation between ranks sharing this node, or 8 ranks
    # each spawning a full-size pool will oversubscribe every core.
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", world_size if is_dist else 1))
    if local_ws > 1 and workers > 0:
        workers = max(0, workers // local_ws)
        if workers == 1:
            workers = 0

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

    # Only rank 0 may touch stdin -- the other ranks have no terminal, and all
    # of them prompting at once would deadlock. Rank 0 decides, everyone obeys.
    if ckpt:
        do_resume = _ask_resume(save_dir, ckpt) if is_main else False
        if is_dist:
            import torch.distributed as dist
            flag = torch.tensor([1 if do_resume else 0], dtype=torch.long)
            dist.broadcast(flag, src=0)
            do_resume = bool(flag.item())
    else:
        do_resume = False

    if do_resume:
        start_epoch    = ckpt["last_epoch"] + 1
        baseline       = ckpt["baseline"]
        global_step    = ckpt["global_step"]
        history        = ckpt["history"]
        history.setdefault("kl_mean", [])
        history["kl_mean"] += [0.0] * (len(history.get("epoch", []))
                                       - len(history["kl_mean"]))
        resume_adapter = os.path.join(save_dir, f"epoch_{ckpt['last_epoch']:03d}")
        log(f"\n  Resuming from epoch {start_epoch} "
            f"(baseline={baseline:.3f}, step={global_step})")
    else:
        log("\n  Starting fresh training run.")

    if start_epoch > num_epochs:
        log("  Training already complete (all epochs done).")
        ddp_cleanup(is_dist)
        return history

    tokenizer, model, device = load_chemberta_for_policy(
        model_name      = model_name,
        lora_rank       = LORA_RANK,
        lora_alpha      = LORA_ALPHA,
        lora_dropout    = LORA_DROPOUT,
        lora_targets    = LORA_TARGET_MODS,
        lora_checkpoint = resume_adapter,
    )

    if kl_beta > 0.0 and KL_BASE_DROPOUT_OFF:
        n_off = disable_base_dropout(model)
        if n_off:
            log(f"  KL anchor: base dropout disabled in {n_off} module(s) "
                f"(config.STAGE9_KL_BASE_DROPOUT_OFF).")

    # ── DDP wrapping ──────────────────────────────────────────────────────
    # `core` stays the unwrapped PeftModel. Everything that is not the
    # gradient-producing forward goes through it: save_pretrained, and above
    # all disable_adapter() for the KL reference. Calling a second forward
    # through the DDP wrapper would register another set of autograd hooks for
    # a pass whose gradients we never want reduced.
    core = model
    if is_dist:
        if torch.cuda.is_available():
            device = f"cuda:{local_rank}"
            model = model.to(device)
            core = model
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=False,
            )
        log(f"  DDP: {world_size} rank(s), backend "
            f"{'nccl' if torch.cuda.is_available() else 'gloo'}.")

    amp_dtype = resolve_amp(device)
    tf32_on   = enable_tf32()
    # fp16 needs loss scaling to keep small gradients from flushing to zero;
    # bf16 has fp32's exponent range and does not.
    scaler = (torch.amp.GradScaler(_device_type(device))
              if amp_dtype == torch.float16 else None)

    est_steps = (len(pairs) // int(batch_size)
                 if not isinstance(batch_size, str) else None)
    tqdm.write(
        f"  7a batched rollout : {'ON' if batched else 'OFF (per-molecule, as Stage 9)'}"
        f"   [config.STAGE9_1_BATCHED_ROLLOUT]\n"
        f"  7b scoring workers : {workers if workers else 'serial (no pool)'}"
        f"   [config.STAGE9_1_SCORING_WORKERS]\n"
        f"  batch size         : {batch_size}"
        + (f"  (~{est_steps} optimizer steps/epoch)" if est_steps else
           f"  (<= {getattr(config, 'STAGE9_1_MAX_BATCH_TOKENS', 65536)} padded tokens/batch)")
        + f"   [config.STAGE9_1_BATCH_SIZE]\n"
        f"  precision          : "
        + (f"{str(amp_dtype).replace('torch.', '')} autocast "
           f"(fp32 log-softmax/KL)" if amp_dtype else "fp32")
        + (f" + TF32 matmul" if tf32_on else "")
        + f"   [config.STAGE9_1_AMP]\n"
        f"  length bucketing   : "
        f"{'ON' if getattr(config, 'STAGE9_1_LENGTH_BUCKETING', True) else 'OFF'}"
        f"   [config.STAGE9_1_LENGTH_BUCKETING]"
    )
    if est_steps is not None and est_steps * num_epochs < 500:
        tqdm.write(
            f"  NOTE: only ~{est_steps * num_epochs} optimizer steps in total. "
            f"REINFORCE is high-variance and needs steps -- consider a smaller "
            f"batch, more epochs, or a proportionally higher learning rate."
        )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    opt_path = os.path.join(save_dir, "optimizer.pt")
    if resume_adapter and os.path.isfile(opt_path):
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            log("  Optimizer state restored.")
        except Exception as e:
            log(f"  Could not restore optimizer state: {e}")

    rng = random.Random(42 + start_epoch)
    remaining_epochs = num_epochs - start_epoch + 1
    # "auto" batching only knows its batch count after packing, so the bar is
    # seeded from the first epoch's split and corrected as epochs are built.
    _first = build_batches(pairs, tokenizer, batch_size, random.Random(0))
    total_batches = remaining_epochs * (len(_first) // max(world_size, 1))

    pbar = tqdm(
        total=total_batches, desc="Stage 9.1 property fine-tuning", unit="batch",
        dynamic_ncols=True, disable=not is_main,
        bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"),
    )

    with ScoringPool(workers) as pool:
        for epoch in range(start_epoch, num_epochs + 1):
            # Every rank builds the identical split from the same seeded rng,
            # then keeps a disjoint, equal-length slice of it.
            batches = build_batches(pairs, tokenizer, batch_size, rng)
            batches = _shard_batches(batches, rank, world_size, is_dist, device)

            ep_reward, ep_loss, ep_valid = [], [], []
            ep_qed, ep_sa, ep_novelty, ep_tox_free, ep_tox21 = [], [], [], [], []
            ep_kl = []

            tqdm.write(f"\n{'-'*60}")
            tqdm.write(f"  Epoch {epoch}/{num_epochs}  ({len(batches)} batches x {batch_size} samples)")
            tqdm.write(f"{'-'*60}")

            for batch in batches:
                optimizer.zero_grad()
                masked_list = [m for m, _ in batch]
                parents     = [o for _, o in batch]
                n           = max(len(batch), 1)

                if batched:
                    log_probs, kls, generated = reinforce_rollout_batched(
                        masked_list, tokenizer, model, device,
                        top_k=top_k, temperature=temperature, kl_beta=kl_beta,
                        amp_dtype=amp_dtype, ref_model=core,
                    )
                else:
                    # Reference path: Stage 9's per-molecule rollout, kept so the
                    # two can be compared on identical data without editing code.
                    lp_list, kl_list, generated = [], [], []
                    for masked_smi, orig_smi in batch:
                        _, lp, kl, gen, _ = reinforce_rollout_oneshot(
                            smiles_masked=masked_smi, tokenizer=tokenizer,
                            model=model, device=device, top_k=top_k,
                            temperature=temperature,
                            reward_fn=lambda s: 0.0, kl_beta=kl_beta,
                        )
                        lp_list.append(lp); kl_list.append(kl); generated.append(gen)
                    log_probs = torch.stack(lp_list)
                    kls       = torch.stack(kl_list)

                scores, comps = score_batch(generated, parents, pool, score_weights)

                # The advantage is a constant multiplier -- no gradient path
                # through the score, which is the whole point of REINFORCE.
                advantage = torch.tensor(scores, device=device, dtype=log_probs.dtype) - baseline
                batch_loss = (-advantage * log_probs).sum() / n
                if kl_beta > 0.0:
                    batch_loss = batch_loss + kl_beta * kls.sum() / n

                if scaler is not None:
                    # unscale_ before clipping so the clip threshold applies to
                    # true gradients, not scaled ones.
                    scaler.scale(batch_loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    batch_loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                if global_step == 0 and torch.cuda.is_available():
                    tqdm.write(
                        f"  GPU memory after first batch: "
                        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak "
                        f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB "
                        f"(batch of {n}, padded to {max(len(m) for m in masked_list)} chars). "
                        f"Raise config.STAGE9_1_BATCH_SIZE if you want more -- but read "
                        f"its note on optimizer steps first."
                    )

                # The baseline enters every rank's advantage, so it must be the
                # SAME number everywhere -- otherwise each rank optimises a
                # slightly different objective and the averaged gradient is not
                # the gradient of anything. Reduce across ranks before updating.
                mean_reward = _all_reduce_mean(sum(scores) / n, is_dist, device)
                baseline = baseline_decay * baseline + (1 - baseline_decay) * mean_reward

                def _avg_c(key):
                    return sum(c[key] for c in comps) / n

                n_masks = sum(m.count(tokenizer.mask_token) for m in masked_list) or 1
                ep_reward.append(mean_reward)
                ep_loss.append(batch_loss.item())
                ep_valid.append(_avg_c("valid"))
                ep_qed.append(_avg_c("qed"))
                ep_sa.append(_avg_c("sa"))
                ep_novelty.append(_avg_c("novelty"))
                ep_tox_free.append(_avg_c("tox_alert"))
                ep_tox21.append(_avg_c("tox21"))
                # Per masked POSITION, as Stage 9 reports it: the summed KL
                # scales with molecule size, so only a per-position figure
                # stays comparable across batches and tunable against beta.
                ep_kl.append(float(kls.sum().detach()) / n_masks if kl_beta > 0 else 0.0)
                global_step += 1

                pbar.set_postfix_str(
                    f"ep={epoch}/{num_epochs}  score={mean_reward:.3f}  "
                    f"loss={batch_loss.item():.4f}  valid={ep_valid[-1]:.0%}  "
                    f"novelty={ep_novelty[-1]:.2f}  tox_free={ep_tox_free[-1]:.0%}  "
                    f"baseline={baseline:.3f}"
                    + (f"  kl/pos={ep_kl[-1]:.3f}" if kl_beta > 0 else ""),
                    refresh=True,
                )
                pbar.update(1)

            def _avg(xs):
                return sum(xs) / max(len(xs), 1)

            log(
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

            # Rank 0 owns the filesystem: 8 ranks writing the same adapter
            # would race, and the weights are identical after all-reduce anyway.
            if is_main:
                epoch_adapter_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
                core.save_pretrained(epoch_adapter_dir)
                _save_checkpoint(
                    save_dir=save_dir, epoch=epoch, baseline=baseline,
                    history=history, optimizer=optimizer, global_step=global_step,
                )
                log(f"  Checkpoint saved -> {epoch_adapter_dir}")
            if is_dist:
                import torch.distributed as dist
                dist.barrier()      # no rank starts the next epoch mid-write

    pbar.close()

    if is_main:
        final_adapter = os.path.join(save_dir, f"epoch_{num_epochs:03d}")
        if os.path.isdir(final_adapter):
            import shutil
            for fname in os.listdir(final_adapter):
                shutil.copy2(os.path.join(final_adapter, fname),
                             os.path.join(save_dir, fname))
        log(f"\n  Final LoRA adapter saved to : {save_dir}")

        from stage9_masked_property_finetune import _plot_stage9_history
        _plot_stage9_history(history, save_dir)

    ddp_cleanup(is_dist)
    return history


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(max_pairs_per_source: int = None, sample_seed: int = None,
         batched: bool = None, workers: int = None) -> None:
    save_dir = getattr(config, "STAGE9_1_LORA_DIR", None) or config.STAGE9_LORA_DIR
    print("\n" + "=" * 60)
    print("STAGE 9.1 -- BATCHED ROLLOUT + PARALLEL RDKit SCORING")
    print("=" * 60)
    print(f"""
  Same objective, data and estimator as Stage 9 -- only faster.
    7a  one padded forward per batch instead of one per molecule
        (config.STAGE9_1_BATCHED_ROLLOUT={getattr(config, 'STAGE9_1_BATCHED_ROLLOUT', 'auto')!r}
         -> {'ON' if resolve_batched(batched) else 'OFF'})
    7b  RDKit scored across a process pool; Tox21 batched on this process
        (config.STAGE9_1_SCORING_WORKERS={getattr(config, 'STAGE9_1_SCORING_WORKERS', 'auto')!r}
         -> {resolve_workers(workers) or 'serial'})

  Output dir : {save_dir}
  Tox21      : {"loaded" if _TOX21_AVAILABLE else "NOT configured -- tox21 term contributes 0"}
""")

    pairs = collect_all_training_pairs()
    if not pairs:
        print("  No training pairs found. Run stage1a/stage1b first.")
        sys.exit(1)

    history = run_stage9_1_finetuning(
        pairs=pairs, save_dir=save_dir, batched=batched, workers=workers,
    )

    print("\n" + "=" * 60)
    print("  Stage 9.1 fine-tuning complete.")
    if history["valid_rate"]:
        print(f"  Final validity rate      : {history['valid_rate'][-1]:.1%}")
        print(f"  Final mean score         : {history['reward_mean'][-1]:.3f}")
        print(f"  Final novelty            : {history['novelty_mean'][-1]:.3f}")
    print(f"  LoRA adapter         : {save_dir}")
    print("=" * 60)

    run_property_distribution_eval(
        lora_checkpoint=save_dir,
        out_path=os.path.join(save_dir, "stage9_1_property_distributions.png"),
        suptitle="Stage 9.1 -- property distributions (batched fine-tuned ChemBERTa)",
        max_pairs_per_source=max_pairs_per_source,
        sample_seed=sample_seed,
    )


def _parse_args(argv: list) -> tuple:
    """
    --limit N / --seed N : as Stage 9 and Stage 9a ("none"/"all"/0 = no limit).
    --workers N          : override config.STAGE9_1_SCORING_WORKERS (0 = serial).
    --no-batch           : force Stage 9's per-molecule rollout (7a off).
    """
    limit = seed = workers = None
    for flag in ("--limit", "--seed", "--workers"):
        if flag not in argv:
            continue
        idx = argv.index(flag)
        if idx + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value, e.g. {flag} 4")
        raw = argv[idx + 1]
        if flag == "--limit":
            limit = 0 if raw.lower() in ("none", "all", "0") else int(raw)
        elif flag == "--seed":
            seed = int(raw)
        else:
            workers = int(raw)
    batched = False if "--no-batch" in argv else None
    return limit, seed, batched, workers


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    """
    The tests that matter for 7a/7b, in order of what they pin down:

      1. padding never reaches real positions (batched logits == batch-1)
      2. the whole batched rollout reproduces Stage 9's EXACTLY at top_k=1,
         where sampling is deterministic -- this is what validates the
         gather / scatter / index_add bookkeeping, not just the forward pass
      3. per-molecule log-prob credit does not bleed across molecules
      4. pooled scoring == serial scoring
      5. a short end-to-end run actually trains
    """
    import tempfile
    from stage9_masked_property_finetune import get_chemberta_tokenizer

    tokenizer, model, device = load_chemberta_for_policy()
    model.eval()

    assert tokenizer.padding_side == "right", (
        "batched rollout requires right padding; left padding shifts RoBERTa "
        "position ids when the pad id disagrees with model.config.pad_token_id"
    )

    # ── 1. padding does not leak into real positions ───────────────────────
    smis = [
        "C<mask>O",
        "CC(=O)O<mask>c1ccccc1",
        "COc1cc2c(cc1OC)C(=O)C(C<mask>1CCN(Cc3ccccc3)CC1)C2",
    ]
    enc = tokenizer(smis, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        batched_logits = model(**enc).logits
    worst = 0.0
    for i, s in enumerate(smis):
        e1 = tokenizer(s, return_tensors="pt").to(device)
        with torch.no_grad():
            single = model(**e1).logits
        n = e1["input_ids"].shape[1]
        worst = max(worst, (batched_logits[i, :n] - single[0]).abs().max().item())
    assert worst < 1e-4, f"padding leaked into real positions: max diff {worst:.2e}"
    assert not torch.isnan(batched_logits).any(), "NaN in padded logits"
    print(f"  [1] padding isolation  : max |batched - single| = {worst:.2e}  OK")

    # ── 2. exact equivalence with Stage 9 at top_k=1 (deterministic) ───────
    # top_k=1 leaves multinomial a single candidate, so both paths must pick
    # the same token at every position. Any indexing bug shows up here.
    lp_b, kl_b, gen_b = reinforce_rollout_batched(
        smis, tokenizer, model, device, top_k=1, temperature=1.0, kl_beta=1.0,
    )
    for i, s in enumerate(smis):
        _, lp_s, kl_s, gen_s, _ = reinforce_rollout_oneshot(
            smiles_masked=s, tokenizer=tokenizer, model=model, device=device,
            top_k=1, temperature=1.0, reward_fn=lambda x: 0.0, kl_beta=1.0,
        )
        assert gen_b[i] == gen_s, f"SMILES differ at top_k=1: {gen_b[i]!r} vs {gen_s!r}"
        assert abs(float(lp_b[i]) - float(lp_s)) < 1e-4, (
            f"log_prob differs at top_k=1: {float(lp_b[i])} vs {float(lp_s)}")
        assert abs(float(kl_b[i]) - float(kl_s)) < 1e-3, (
            f"KL differs at top_k=1: {float(kl_b[i])} vs {float(kl_s)}")
    print("  [2] batched == Stage 9 at top_k=1 (SMILES, log_prob, KL)  OK")

    # ── 3. gradients flow, and credit stays per molecule ───────────────────
    model.train()
    lp, kl, gen = reinforce_rollout_batched(
        smis, tokenizer, model, device, top_k=5, kl_beta=0.05,
    )
    assert lp.shape == (len(smis),) and kl.shape == (len(smis),)
    assert lp.requires_grad, "log_probs must carry a gradient"
    # A molecule with no mask must contribute exactly zero.
    lp0, _, gen0 = reinforce_rollout_batched(
        ["CCO", "C<mask>O"], tokenizer, model, device, top_k=5,
    )
    assert float(lp0[0]) == 0.0, "unmasked molecule must have zero log_prob"
    assert float(lp0[1]) != 0.0, "masked molecule must have nonzero log_prob"
    print("  [3] per-molecule credit + grad flow  OK")

    # ── 4. pooled scoring == serial scoring ────────────────────────────────
    gen_list = ["CC(=O)Oc1ccccc1", "not_a_smiles(((", "CCO"]
    par_list = ["CC(=O)Oc1ccccc1", "CCO", "CCO"]
    with ScoringPool(0) as serial:
        s_ser, c_ser = score_batch(gen_list, par_list, serial)
    n_workers = resolve_workers()
    if n_workers > 0:
        with ScoringPool(n_workers) as par:
            s_par, c_par = score_batch(gen_list, par_list, par)
        for a, b in zip(s_ser, s_par):
            assert abs(a - b) < 1e-9, f"pooled score {b} != serial {a}"
        print(f"  [4] pooled ({n_workers} workers) == serial scoring  OK")
    else:
        print("  [4] pooled scoring skipped (auto resolved to serial)  OK")
    assert c_ser[1]["valid"] == 0.0, "invalid SMILES must score valid=0"
    assert c_ser[0]["novelty"] == 0.0, "identical molecule must score novelty=0"

    # ── 4b. batching: no pair lost or duplicated, budgets respected ────────
    many = [(f"C{'C'*i}<mask>O", f"C{'C'*i}CO") for i in range(1, 120)]
    for bs in (16, "auto"):
        bat = build_batches(many, tokenizer, bs, random.Random(0))
        flat = [p for b in bat for p in bat and b]
        assert sorted(flat) == sorted(many), f"batching lost/duplicated pairs at {bs}"
        if bs == "auto":
            budget = getattr(config, "STAGE9_1_MAX_BATCH_TOKENS", 65536)
            for b in bat:
                lens = _token_lengths([m for m, _ in b], tokenizer)
                assert len(b) * max(lens) <= budget or len(b) == 1, (
                    f"auto batch exceeded token budget: {len(b)}x{max(lens)}")
        else:
            assert all(len(b) <= bs for b in bat)
    # Bucketing must actually reduce padding waste on a length-varied set.
    def _waste(bs_list):
        tot = pad = 0
        for b in bs_list:
            lens = _token_lengths([m for m, _ in b], tokenizer)
            tot += sum(lens); pad += len(b) * max(lens)
        return pad / tot
    w_on  = _waste(build_batches(many, tokenizer, 16, random.Random(0), bucketing=True))
    w_off = _waste(build_batches(many, tokenizer, 16, random.Random(0), bucketing=False))
    assert w_on < w_off, f"bucketing did not reduce padding ({w_on:.2f} vs {w_off:.2f})"
    print(f"  [4b] batching intact; padding overhead {w_off:.2f}x -> {w_on:.2f}x bucketed  OK")

    # ── 4c. DDP helpers, single-process ────────────────────────────────────
    # The full multi-rank run lives in stage9_1_ddp_smoketest.py (needs >= 2
    # processes). What is checkable here is the logic those ranks rely on --
    # and getting it wrong is what causes silent divergence or a hang, so it
    # is worth pinning even without a process group.
    assert ddp_setup() == (0, 1, 0, False), "must fall back to single-process cleanly"
    fake = [[("m", "p")] for _ in range(10)]
    assert _shard_batches(fake, 0, 1, False, "cpu") == fake, "no-op when not distributed"
    assert _all_reduce_mean(0.75, False, "cpu") == 0.75
    # Disjoint, equal-length shards: the property that keeps DDP from hanging.
    for ws in (2, 3, 4):
        shards = [fake[r::ws] for r in range(ws)]
        n = min(len(x) for x in shards)
        shards = [x[:n] for x in shards]
        assert len({id(b) for s in shards for b in s}) == ws * n, "shards overlap"
        assert len({len(s) for s in shards}) == 1, "ranks would run unequal steps -> hang"
    print("  [4c] DDP helpers: single-process fallback + disjoint equal shards  OK")

    # ── 5. short end-to-end run ────────────────────────────────────────────
    test_pairs = [
        ("CC(=O)OC1=CC=CC=C1C(=O)<mask>", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)<mask>", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)<mask>", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ("CC(=O)NC1=CC=C(<mask>)C=C1", "CC(=O)NC1=CC=C(O)C=C1"),
    ]
    with tempfile.TemporaryDirectory() as td:
        hist = run_stage9_1_finetuning(
            pairs=list(test_pairs), save_dir=td, num_epochs=1, batch_size=2,
            top_k=5, batched=True, workers=0,
        )
        assert hist["epoch"] == [1] and len(hist["reward_mean"]) == 1
        assert os.path.isdir(os.path.join(td, "epoch_001"))
    print("  [5] end-to-end batched training run  OK")

    print("Stage 9.1 self-test passed.")


if __name__ == "__main__":
    # Required before any pool is created: under "spawn" the children re-import
    # this module, and without the guard they would re-run training recursively.
    mp.freeze_support()
    if "--hardware" in sys.argv:
        from hardware_autotune import get_profile as _gp
        print(_gp().describe())
        sys.exit(0)
    if "--test" in sys.argv:
        _run_self_test()
    else:
        _limit, _seed, _batched, _workers = _parse_args(sys.argv)
        main(max_pairs_per_source=_limit, sample_seed=_seed,
             batched=_batched, workers=_workers)

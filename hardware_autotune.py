# -*- coding: utf-8 -*-
"""
hardware_autotune.py
====================
One place that answers "what machine am I on, and how hard can I drive it?"

Shared by the compute-heavy stages (Stage 1 mask calculation, Stage 9, Stage
9a, Stage 9.1) so a run adapts to wherever it lands -- a 2-vCPU Colab T4, a
Colab L4, or an HPC job holding an A100 or H100 -- without any per-machine
edits to config.py.

Why this is not just os.cpu_count() and torch.cuda.is_available()
------------------------------------------------------------------
Both of those LIE in exactly the environments this project runs in.

  os.cpu_count() reports the HOST's cores, not what your job was allocated. On
  a shared HPC node with 128 cores, a job granted 8 will still see 128, spawn
  128 workers, and thrash -- while the scheduler accounts it as using 8. The
  allocation is in SLURM_CPUS_PER_TASK, in the process's CPU affinity mask, or
  in a cgroup quota, and all three are checked here in that order.

  Container memory limits are invisible to psutil.virtual_memory(), which
  reports host RAM. A cgroup memory cap is what actually kills your process, so
  it is read directly.

Getting these wrong is not a missed optimisation, it is a job that gets killed
or one that oversubscribes a shared node.

What it decides
---------------
  cpu_workers        processes for RDKit/PLIP pools (allocation-aware)
  torch_threads      intra-op threads, and 1-per-worker inside pools so N
                     workers x M BLAS threads never exceeds the allocation
  amp_dtype          bf16 on Ampere+/Hopper (A100 cc8.0, L4 cc8.9, H100 cc9.0),
                     fp16 on Turing/Volta (T4 cc7.5, V100 cc7.0), else None
  tf32               on for cc >= 8.0
  batch size         from free GPU memory, rounded to a multiple of 8 so the
                     matmuls stay tensor-core aligned

On MAC utilisation specifically: tensor cores need three things, and all three
are handled -- a supported dtype (bf16/fp16 via amp_dtype), TF32 for the fp32
matmuls that remain, and shapes that are multiples of 8. A batch of 63 runs the
same kernels as 64 while leaving a lane idle, so batch suggestions are always
rounded.

What it will NOT pretend to do
-------------------------------
  MULTI-GPU: the count is detected and reported, but every training loop in
  this project is single-device. Using 8 H100s needs DistributedDataParallel
  and a rewritten training step; until that exists, the report says how many
  GPUs are visible and that only one is used, rather than quietly implying a
  speedup that is not happening.

  TPU: presence of torch_xla is detected and reported, but ChemBERTa + peft
  training here has no XLA path. Reported as unavailable, not silently ignored.

Usage
-----
  python hardware_autotune.py            # print the report and exit

  from hardware_autotune import get_profile
  hw = get_profile()
  hw.apply()                             # set torch threads / TF32 / precision
  print(hw.describe())
  pool_size = hw.cpu_workers
  dtype     = hw.amp_dtype
  bs        = hw.recommended_batch_size(seq_len=128, vocab=767)
"""

from __future__ import annotations

import os
import platform
import sys
from typing import List, Optional

try:
    import torch
    _TORCH = True
except ImportError:                                  # pragma: no cover
    _TORCH = False

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

_GIB = float(2 ** 30)


# ════════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL PROBES
# ════════════════════════════════════════════════════════════════════════════

def _cgroup_cpu_quota() -> Optional[float]:
    """CPU quota in cores from cgroup v2 then v1, or None if uncapped."""
    try:                                             # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return float(quota) / float(period)
    except Exception:
        pass
    try:                                             # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            return quota / period
    except Exception:
        pass
    return None


def _cgroup_mem_limit_gib() -> Optional[float]:
    """Container memory cap in GiB, or None if uncapped/absent."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw == "max":
                continue
            val = int(raw)
            # cgroup v1 writes a sentinel near 2**63 to mean "no limit".
            if 0 < val < 2 ** 62:
                return val / _GIB
        except Exception:
            continue
    return None


def detect_cpus() -> tuple:
    """
    (usable_cores, source) -- how many cores this PROCESS may actually use.

    Precedence is deliberate: an explicit scheduler allocation beats an
    affinity mask, which beats a cgroup quota, which beats the host count.
    Every earlier source is a statement about this job; the last is a
    statement about the machine, and using it on a shared node is how you
    oversubscribe.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.environ.get(var)
        if raw and raw.isdigit() and int(raw) > 0:
            return int(raw), var

    if hasattr(os, "sched_getaffinity"):             # Linux only
        try:
            n = len(os.sched_getaffinity(0))
            if n > 0:
                quota = _cgroup_cpu_quota()
                if quota and quota < n:
                    return max(1, int(quota)), "cgroup cpu quota"
                return n, "sched_getaffinity"
        except Exception:
            pass

    quota = _cgroup_cpu_quota()
    if quota:
        return max(1, int(quota)), "cgroup cpu quota"

    return max(1, os.cpu_count() or 1), "os.cpu_count"


def detect_memory() -> tuple:
    """(total_gib, available_gib, source) for the limit that will actually kill us."""
    cg = _cgroup_mem_limit_gib()
    if _PSUTIL:
        vm = psutil.virtual_memory()
        total, avail = vm.total / _GIB, vm.available / _GIB
        if cg and cg < total:
            return cg, min(avail, cg), "cgroup memory limit"
        return total, avail, "psutil"
    if cg:
        return cg, cg, "cgroup memory limit"
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = float(v.strip().split()[0]) / (1024 * 1024)
        return info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0), "/proc/meminfo"
    except Exception:
        return 0.0, 0.0, "unknown"


def detect_tpu() -> Optional[str]:
    """TPU description if torch_xla is importable AND a device is present."""
    try:
        import torch_xla.core.xla_model as xm        # type: ignore
        return str(xm.xla_device())
    except Exception:
        return None


def detect_environment() -> str:
    """Best-effort label for where this is running."""
    if os.environ.get("SLURM_JOB_ID"):
        return (f"SLURM job {os.environ['SLURM_JOB_ID']}"
                f"{' on ' + os.environ['SLURM_JOB_NODELIST']
                   if os.environ.get('SLURM_JOB_NODELIST') else ''}")
    if os.environ.get("PBS_JOBID"):
        return f"PBS job {os.environ['PBS_JOBID']}"
    # The /content fallback is Linux-only on purpose: on Windows a leading
    # slash resolves against the current drive, so "/content" silently becomes
    # "C:\content" -- and any machine that happens to have that folder would
    # otherwise be reported as Colab.
    if "google.colab" in sys.modules or (
        platform.system() == "Linux" and os.path.isdir("/content")
    ):
        return "Google Colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "Kaggle"
    return f"{platform.system()} {platform.release()}"


# ════════════════════════════════════════════════════════════════════════════
#  PROFILE
# ════════════════════════════════════════════════════════════════════════════

class HardwareProfile:
    """Detected hardware plus the settings derived from it. Build via get_profile()."""

    def __init__(self) -> None:
        self.environment = detect_environment()
        self.cpu_count, self.cpu_source = detect_cpus()
        self.ram_total_gib, self.ram_avail_gib, self.ram_source = detect_memory()
        self.tpu = detect_tpu()

        self.gpus: List[dict] = []
        self.device = "cpu"
        if _TORCH and torch.cuda.is_available():
            self.device = "cuda"
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                self.gpus.append({
                    "index": i,
                    "name": p.name,
                    "total_gib": p.total_memory / _GIB,
                    "capability": (p.major, p.minor),
                    "sm_count": getattr(p, "multi_processor_count", None),
                })

    # ── capability flags ──────────────────────────────────────────────────
    @property
    def capability(self) -> tuple:
        return self.gpus[0]["capability"] if self.gpus else (0, 0)

    @property
    def has_tensor_cores(self) -> bool:
        """Volta (7.0) and later. Below that, mixed precision buys little."""
        return self.capability[0] >= 7

    @property
    def supports_bf16(self) -> bool:
        """Ampere (8.0) and later. bf16 keeps fp32's exponent range, so no scaler."""
        if not self.gpus:
            return False
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return self.capability[0] >= 8

    @property
    def supports_tf32(self) -> bool:
        return self.capability[0] >= 8

    @property
    def amp_dtype(self):
        """
        bf16 on Ampere+/Hopper, fp16 on Volta/Turing, None otherwise.

        fp16 needs a GradScaler (its exponent range underflows small gradients);
        bf16 does not. Callers must branch on `needs_grad_scaler`.
        """
        if not _TORCH or not self.gpus:
            return None
        if self.supports_bf16:
            return torch.bfloat16
        if self.has_tensor_cores:
            return torch.float16
        return None

    @property
    def needs_grad_scaler(self) -> bool:
        return _TORCH and self.amp_dtype == torch.float16

    # ── derived settings ──────────────────────────────────────────────────
    @property
    def cpu_workers(self) -> int:
        """
        Processes for a CPU-bound pool (RDKit, PLIP, obabel).

        One core is reserved for the parent, which still drives the GPU and the
        tokenizer. Returns 0 rather than 1 for a single worker: a one-worker
        pool pays pickling and IPC for no parallelism and is strictly slower
        than inline work. Capped at 16 -- past that these workloads are bound
        by memory bandwidth and pool IPC, not cores.
        """
        n = self.cpu_count - 1
        if n <= 1:
            return 0
        return min(n, 16)

    @property
    def torch_threads(self) -> int:
        """Intra-op threads when NOT inside a pool. Never exceeds the allocation."""
        return max(1, self.cpu_count)

    def memory_ceiling_batch(
        self,
        seq_len:        int   = 128,
        vocab:          int   = 767,
        headroom:       float = 0.55,
        bytes_per_elem: int   = None,
        max_batch:      int   = 4096,
    ) -> int:
        """
        Largest batch that FITS -- the memory ceiling, not a recommendation.

        For a masked-LM rollout the dominant tensors are the [B, L, V] logits:
        the policy's (kept for backward), its gradient, the frozen reference's,
        and the activations behind them. `_MULT` is that multiplier -- an
        ESTIMATE, deliberately conservative; `headroom` leaves the rest of the
        card for fragmentation and optimizer state.

        An estimate, not a measurement: Stage 9.1 prints real peak usage after
        its first batch, and that number wins over this one.
        """
        if not self.gpus:
            return 64
        if bytes_per_elem is None:
            bytes_per_elem = 2 if self.amp_dtype is not None else 4

        _MULT = 6
        free_bytes = self.gpus[0]["total_gib"] * _GIB * headroom
        per_sample = seq_len * vocab * bytes_per_elem * _MULT
        return max(8, min(int(free_bytes // max(per_sample, 1)), max_batch))

    def recommended_batch_size(
        self,
        seq_len:         int = 128,
        vocab:           int = 767,
        n_pairs:         int = None,
        n_epochs:        int = None,
        min_total_steps: int = 2000,
        max_batch:       int = 512,
        min_batch:       int = 8,
        **kw,
    ) -> int:
        """
        Batch size to actually use -- which is NOT the memory ceiling.

        The measured result on every card this project targets is that memory
        is not the binding constraint: ChemBERTa is 44M parameters, so a T4 and
        an H100 alike could hold thousands of samples at L=128. Sizing to fill
        the card would be sizing to the wrong number.

        What binds is OPTIMIZER STEPS. Steps = n_pairs / batch * n_epochs, and
        REINFORCE is a high-variance estimator that needs them; doubling the
        batch halves the learning for the same data. So when `n_pairs` and
        `n_epochs` are supplied, the batch is capped to keep at least
        `min_total_steps` updates -- the run stays fast AND still learns.
        Without them it falls back to `max_batch`, which is a throughput
        default rather than a memory one.

        Rounded DOWN to a multiple of 8 so GEMM tiles stay tensor-core aligned:
        a batch of 63 issues the same work as 64 with a lane sitting idle.
        """
        ceiling = self.memory_ceiling_batch(seq_len=seq_len, vocab=vocab, **kw)
        b = min(ceiling, max_batch)

        if n_pairs and n_epochs:
            step_limited = max(1, (n_pairs * n_epochs) // max(min_total_steps, 1))
            b = min(b, step_limited)

        b = (b // 8) * 8
        return max(min_batch, b)

    # ── application ───────────────────────────────────────────────────────
    def apply(self, in_worker: bool = False) -> List[str]:
        """
        Push the derived settings into torch. Returns what changed, for logging.

        `in_worker=True` pins BLAS to a single thread: N pool workers each
        spawning M threads on an N-core allocation is the classic
        oversubscription that makes a parallel run slower than a serial one.
        """
        applied: List[str] = []
        if not _TORCH:
            return applied

        threads = 1 if in_worker else self.torch_threads
        try:
            torch.set_num_threads(threads)
            applied.append(f"torch threads = {threads}")
        except Exception:
            pass
        if in_worker:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[var] = "1"
            applied.append("BLAS threads pinned to 1 (worker)")

        if self.supports_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                applied.append("TF32 matmul enabled (cc >= 8.0)")
            except Exception:
                pass
        return applied

    # ── reporting ─────────────────────────────────────────────────────────
    def describe(self) -> str:
        L: List[str] = []
        add = L.append
        add("=" * 68)
        add("  HARDWARE PROFILE")
        add("=" * 68)
        add(f"  Environment : {self.environment}")
        add(f"  Python      : {platform.python_version()}"
            + (f"   torch {torch.__version__}" if _TORCH else "   torch MISSING"))
        add("")
        add(f"  CPU cores usable : {self.cpu_count}   (source: {self.cpu_source})")
        if self.cpu_source != "os.cpu_count":
            host = os.cpu_count() or 0
            if host and host != self.cpu_count:
                add(f"      host reports {host} -- using the allocation, not the host")
        add(f"  RAM              : {self.ram_total_gib:.1f} GiB total, "
            f"{self.ram_avail_gib:.1f} GiB available   (source: {self.ram_source})")
        add("")
        if self.gpus:
            for g in self.gpus:
                cc = f"{g['capability'][0]}.{g['capability'][1]}"
                add(f"  GPU {g['index']} : {g['name']}  "
                    f"{g['total_gib']:.1f} GiB  compute capability {cc}"
                    + (f"  {g['sm_count']} SMs" if g["sm_count"] else ""))
            if len(self.gpus) > 1:
                add(f"      {len(self.gpus)} GPUs visible, but this project's training "
                    f"loops are SINGLE-DEVICE.")
                add(f"      Only GPU 0 will be used. Multi-GPU needs "
                    f"DistributedDataParallel, which is not implemented.")
        else:
            add("  GPU : none detected -- running on CPU")
        if self.tpu:
            add(f"  TPU : {self.tpu} detected, but no XLA path exists in these "
                f"stages -- it will NOT be used.")
        add("")
        add("  DERIVED SETTINGS")
        add(f"    pool workers    : {self.cpu_workers or 'serial (no pool)'}")
        add(f"    torch threads   : {self.torch_threads}")
        dt = self.amp_dtype
        add(f"    precision       : "
            + (f"{str(dt).replace('torch.', '')} autocast"
               + ("  + GradScaler" if self.needs_grad_scaler else "  (no scaler needed)")
               if dt is not None else "fp32"))
        add(f"    TF32 matmul     : {'yes' if self.supports_tf32 else 'no'}")
        add(f"    tensor cores    : {'yes' if self.has_tensor_cores else 'no'}")
        add(f"    memory ceiling  : ~{self.memory_ceiling_batch(seq_len=128)} samples "
            f"@ L=128 (estimate)")
        add(f"    batch suggested : {self.recommended_batch_size(seq_len=128)}"
            f"   <- throughput/steps, NOT memory. This model cannot fill")
        add(f"                      these cards; sizing to memory would cost "
            f"optimizer steps.")
        add("=" * 68)
        return "\n".join(L)


_PROFILE: Optional[HardwareProfile] = None


def get_profile(refresh: bool = False) -> HardwareProfile:
    """The cached profile. Detection touches /sys and torch, so it is done once."""
    global _PROFILE
    if _PROFILE is None or refresh:
        _PROFILE = HardwareProfile()
    return _PROFILE


# ════════════════════════════════════════════════════════════════════════════
#  CPU-BOUND HELPER  (Stage 1 mask calculation, RDKit scoring, ...)
# ════════════════════════════════════════════════════════════════════════════

def parallel_map(fn, items, workers: int = None, chunksize: int = None,
                 initializer=None, desc: str = None):
    """
    Allocation-aware process-pool map for CPU-bound work, with a serial
    fallback that keeps the exact same contract.

    Uses the "spawn" context explicitly so behaviour matches on Linux, Windows
    and Colab rather than forking on one and spawning on another -- fork copies
    a CUDA context into the child and corrupts it, which is a genuinely
    unpleasant bug to chase.

    `fn` must be a module-level function (spawn pickles it by qualified name;
    a lambda or closure will not survive).
    """
    items = list(items)
    if not items:
        return []
    if workers is None:
        workers = get_profile().cpu_workers
    if workers <= 0:
        return [fn(x) for x in items]

    import multiprocessing as mp
    if chunksize is None:
        chunksize = max(1, len(items) // (workers * 4) or 1)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=initializer) as pool:
        return pool.map(fn, items, chunksize=chunksize)


def worker_init() -> None:
    """
    Default pool initializer: pin BLAS to one thread per worker.

    Pass as `initializer` to any pool built here. Without it, W workers each
    starting T BLAS threads produce W*T threads on a W-core allocation, and the
    contention typically makes the parallel version slower than serial.
    """
    get_profile().apply(in_worker=True)


if __name__ == "__main__":
    hw = get_profile()
    print(hw.describe())
    changed = hw.apply()
    if changed:
        print("  Applied: " + "; ".join(changed))

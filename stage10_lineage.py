# -*- coding: utf-8 -*-
"""
stage10_lineage.py
==================
Where the Stage 10 family writes, and what one of its runs may inherit from
another.

The three Stage 10 scripts train the SAME objective on the SAME pairs and
differ only in how that objective is executed:

    stage10_vanila_backpropagation_training.py    reference, per-molecule, fp32
    stage10_1_parallel_RDKit_scoring_...py        + pooled RDKit, step-resumable
    stage10_2_hardware_tuned_...py                + batched forward, AMP, DDP

By default each writes to its own directory and resumes only its own
checkpoints, which is what keeps a 10-vs-10.1-vs-10.2 timing comparison a
comparison of execution and nothing else.

config.STAGE10_SHARED_OUTPUT = True switches all three to ONE directory per
variant and ONE rolling checkpoint, so a run interrupted under any of them can
be continued under any other -- start on the reference implementation, move to
the tuned one when a GPU frees up, finish wherever. That is a genuinely useful
mode and a genuinely dangerous one, and this module exists so that both halves
are handled in one place rather than three.

WHAT "ONE CHECKPOINT" HAS TO CARRY
-----------------------------------
Weights alone are not enough to continue a run. Stage 10's own per-epoch format
saves only the unfrozen tensors, so continuing from it rebuilds Adam from
scratch and discards exp_avg / exp_avg_sq -- the entire adaptive state that
makes Adam Adam. Stage 10.1 already solved that for itself; the lineage
checkpoint here is that format, promoted to the family's shared one:

    trainable      the unfrozen tensors (~31 MB)
    optimizer      Adam's moments (~62 MB)
    epoch          the NEXT epoch to run
    batch_index    the NEXT batch within it (0 = epoch boundary)
    global_step    optimizer steps taken so far
    history        the per-epoch diagnostics list, so the curve is continuous
    agg / n_steps  running mid-epoch aggregates, so a resume does not compute
                   an epoch's history row from only the batches after restart
    rng            python + torch (+ cuda) streams
    fingerprint    variant / batch_size / k_cand / num_epochs / n_pairs / seed
    provenance     WHICH STAGE trained WHICH EPOCHS, under what execution

WHY PROVENANCE IS NOT OPTIONAL HERE
------------------------------------
A lineage is allowed to mix execution paths -- that is the point of the mode --
but mixing them changes what the run IS. Stage 10 and 10.1 sample candidates
per molecule in fp32; Stage 10.2 can sample the whole batch in one call under
bfloat16 autocast. Both are the same objective in expectation, and neither is
wrong, but a model whose epochs 1-2 came from one and 3-4 from the other is not
the controlled run either script describes on its own.

Nothing on disk would record that. `model.safetensors` has no memory of the
autocast dtype it was trained under, and by the time anyone asks, the terminal
scrollback is gone. So every save appends to a provenance list, every
cross-stage or cross-precision resume prints a banner naming exactly what
changed, and the JSON sidecar keeps the whole table in plain text next to the
weights -- which is what makes the honest sentence writable months later.

Provenance is recorded as contiguous SEGMENTS rather than one row per save: a
thousand checkpoints written by one stage under one setting are one row, and a
new row appears only when the execution actually changes. The table stays
readable at a glance, which is the only way it gets read.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Dict, List, Mapping, Optional

import config

try:
    from tqdm import tqdm as _tqdm
except ImportError:                                  # pragma: no cover
    # Never None: tqdm_compat provides the same surface, tqdm.write included.
    from tqdm_compat import tqdm as _tqdm

# Bumped from Stage 10.1's private format 1, which had no provenance list.
# Format 1 checkpoints are still readable -- see load_state.
CKPT_FORMAT = 2

STAGES = ("stage10", "stage10_1", "stage10_2", "stage10_4")

# The stages sharing ONE objective, and so the only ones a shared checkpoint
# may pass between. Stage 10, 10.1 and 10.2 differ purely in execution -- the
# loss they minimise is the same function of the same measurements -- so a run
# begun under one can legitimately be finished under another.
#
# Stage 10.4 is NOT in here. It adds a fifth loss term (the Tox21 classifier),
# which moves its LOSS_INVALID from 2.00 to 2.20 and changes which candidate
# wins best-of-K. Handing it a Stage 10.2 checkpoint would be training one
# model on two objectives and reporting the result as one run, and the epochs
# already in the history would be plotted on the same axes as epochs that
# optimised something else. shared_enabled() therefore returns False for it
# whatever config says, so it always reads and writes its own directory and
# its own checkpoint name.
LINEAGE_STAGES = ("stage10", "stage10_1", "stage10_2")

# In shared mode every stage reads and writes these two names, which is what
# makes a checkpoint written by one stage findable by the next.
LINEAGE_CKPT_NAME = "stage10_lineage.pt"
LINEAGE_META_NAME = "stage10_lineage.json"

# In private mode each stage keeps the filename it already used, so switching
# the flag off does not orphan an existing run's checkpoint.
_PRIVATE_NAMES: Dict[str, tuple] = {
    "stage10":   ("stage10_checkpoint.pt",   "stage10_checkpoint.json"),
    "stage10_1": ("stage10_1_checkpoint.pt", "stage10_1_checkpoint.json"),
    "stage10_2": ("stage10_2_checkpoint.pt", "stage10_2_checkpoint.json"),
    "stage10_4": ("stage10_4_checkpoint.pt", "stage10_4_checkpoint.json"),
}

# The config entry holding each stage's OWN output directory, used when the
# shared flag is off.
_PRIVATE_DIR_KEYS: Dict[str, tuple] = {
    "stage10":   ("STAGE10_DIR",   "./outputs/stage10/"),
    "stage10_1": ("STAGE10_1_DIR", "./outputs/stage10_1/"),
    "stage10_2": ("STAGE10_2_DIR", "./outputs/stage10_2/"),
    "stage10_4": ("STAGE10_4_DIR", "./outputs/stage10_4/"),
}

# Set by --shared / --separate on any of the three command lines. Process-wide,
# so a single run can be pointed at the other mode without editing config --
# the same convention as Stage 9.1's --speed.
_SHARED_OVERRIDE: Optional[bool] = None


# ════════════════════════════════════════════════════════════════════════════
#  THE FLAG
# ════════════════════════════════════════════════════════════════════════════

def set_shared_override(value: Optional[bool]) -> None:
    """CLI override for config.STAGE10_SHARED_OUTPUT. None clears it."""
    global _SHARED_OVERRIDE
    _SHARED_OVERRIDE = None if value is None else bool(value)


def shared_enabled(override: Optional[bool] = None,
                   stage: Optional[str] = None) -> bool:
    """
    Is the family sharing one output directory and one checkpoint?

    Precedence, highest first: the stage's own eligibility, then an explicit
    argument, then --shared/--separate, then config.STAGE10_SHARED_OUTPUT,
    which defaults to False. False is the default deliberately: separate
    directories are what make the stages a controlled comparison, and turning
    that off should be a decision someone made rather than one they inherited.

    A stage outside LINEAGE_STAGES is NEVER shared, and that check comes first
    -- ahead of the explicit override, not behind it. Sharing is only sound
    between stages that minimise the same function; for one that does not, an
    explicit shared=True is not a preference to honour but a request that
    would silently mix two objectives into one checkpoint. Passing `stage` is
    optional so the existing three-argument call sites keep working, but every
    call site inside this module that knows its stage passes it.
    """
    if stage is not None and stage not in LINEAGE_STAGES:
        return False
    if override is not None:
        return bool(override)
    if _SHARED_OVERRIDE is not None:
        return _SHARED_OVERRIDE
    return bool(getattr(config, "STAGE10_SHARED_OUTPUT", False))


def _check_stage(stage: str) -> str:
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return stage


# ════════════════════════════════════════════════════════════════════════════
#  WHERE THINGS LAND
# ════════════════════════════════════════════════════════════════════════════

def base_dir(stage: str, shared: Optional[bool] = None) -> str:
    """The stage's output root, before the per-variant subdirectory."""
    _check_stage(stage)
    if shared_enabled(shared, stage):
        return getattr(config, "STAGE10_SHARED_DIR",
                       "./outputs/stage10_family_shared/")
    key, fallback = _PRIVATE_DIR_KEYS[stage]
    return getattr(config, key, None) or fallback


def resolve_save_dir(stage: str, variant: str,
                     shared: Optional[bool] = None) -> str:
    """
    Where this (stage, variant) writes.

    The variant subdirectory survives shared mode. 10a and 10b are DIFFERENT
    objectives -- 10a adds the unlikelihood term -- so merging them would not
    be continuing one run, it would be training one model on two losses. Only
    the execution is allowed to vary along a lineage; the objective is not.
    """
    return os.path.join(base_dir(stage, shared), f"variant_{str(variant).lower()}")


def ckpt_path(save_dir: str, stage: str, shared: Optional[bool] = None) -> str:
    """Rolling checkpoint path: the lineage name when shared, else the
    stage's own."""
    _check_stage(stage)
    name = (LINEAGE_CKPT_NAME if shared_enabled(shared, stage)
            else _PRIVATE_NAMES[stage][0])
    return os.path.join(save_dir, name)


def meta_path(save_dir: str, stage: str, shared: Optional[bool] = None) -> str:
    """The plain-text sidecar next to the checkpoint -- same content minus the
    tensors, so progress and provenance can be read with `cat` rather than by
    loading 93 MB of torch state."""
    _check_stage(stage)
    name = (LINEAGE_META_NAME if shared_enabled(shared, stage)
            else _PRIVATE_NAMES[stage][1])
    return os.path.join(save_dir, name)


# ════════════════════════════════════════════════════════════════════════════
#  EXECUTION PROVENANCE
# ════════════════════════════════════════════════════════════════════════════

# The fields that decide whether two segments are "the same execution". Worker
# counts, checkpoint cadence and device index are recorded but deliberately NOT
# compared: they change wall-clock, never the arithmetic or the sampling.
_EXEC_KEYS = ("stage", "amp", "batched", "speed")


def execution_profile(stage: str, *, speed: Optional[str] = None,
                      amp: Optional[str] = None, batched: bool = False,
                      workers: int = 0, device: str = "cpu",
                      world_size: int = 1, **extra) -> Dict[str, object]:
    """
    A one-line description of HOW a run is executing, for the provenance log.

    `amp` is the autocast dtype as a short string ("bf16"/"fp16") or None for
    full fp32 -- a string rather than a torch.dtype so the JSON sidecar can
    hold it. `batched` says whether candidate sampling drew the whole batch in
    one call (Stage 10.2's fast path) or one molecule at a time (everything
    else); it is the field that decides whether two runs drew the SAME
    candidates, which is what selects the training targets.
    """
    prof: Dict[str, object] = {
        "stage": _check_stage(stage), "speed": speed, "amp": amp,
        "batched": bool(batched), "workers": int(workers),
        "device": str(device), "world_size": int(world_size),
    }
    prof.update(extra)
    return prof


def execution_differs(previous: Dict[str, object],
                      current: Dict[str, object]) -> List[str]:
    """Which of the meaningful execution fields changed. Empty = same path."""
    if not previous:
        return []
    return [k for k in _EXEC_KEYS if previous.get(k) != current.get(k)]


def append_segment(provenance: Optional[List[dict]], profile: Dict[str, object],
                   epoch: int, global_step: int) -> List[dict]:
    """
    Fold this save into the provenance log.

    Extends the last segment when the execution is unchanged, and opens a new
    one when it is not -- so the table is a short list of "who trained what",
    not one row per checkpoint write.
    """
    prov = list(provenance or [])
    now = time.time()
    if prov and not execution_differs(prov[-1], profile):
        prov[-1]["to_epoch"] = epoch
        prov[-1]["to_step"] = global_step
        prov[-1]["updated_at"] = now
        return prov
    seg = dict(profile)
    seg.update({"from_epoch": epoch, "from_step": global_step,
                "to_epoch": epoch, "to_step": global_step,
                "started_at": now, "updated_at": now})
    prov.append(seg)
    return prov


def _describe_exec(p: Dict[str, object]) -> str:
    """One human-readable line for a provenance segment or a live profile."""
    amp = p.get("amp") or "fp32"
    sampling = "batched sampling" if p.get("batched") else "per-molecule sampling"
    bits = [str(p.get("stage", "?")), amp, sampling]
    if p.get("speed"):
        bits.insert(1, f"speed={p['speed']}")
    if int(p.get("world_size", 1) or 1) > 1:
        bits.append(f"{p['world_size']} ranks")
    return ", ".join(bits)


def provenance_table(provenance: Optional[List[dict]]) -> List[str]:
    """The provenance log as printable lines, oldest first."""
    if not provenance:
        return ["    (none recorded)"]
    out = []
    for seg in provenance:
        span = (f"epoch {seg.get('from_epoch', '?')}"
                if seg.get("from_epoch") == seg.get("to_epoch")
                else f"epochs {seg.get('from_epoch', '?')}-{seg.get('to_epoch', '?')}")
        steps = f"{seg.get('from_step', '?')}-{seg.get('to_step', '?')}"
        out.append(f"    {span:<16} steps {steps:<14}  {_describe_exec(seg)}")
    return out


def lineage_banner(ckpt: Optional[dict],
                   profile: Dict[str, object]) -> Optional[str]:
    """
    The warning to print when resuming a checkpoint another execution path
    wrote, or None when the path is unchanged.

    Loud on purpose. This is the moment a run stops being the thing its own
    script's docstring describes, and it is the only moment at which anyone is
    in a position to notice.
    """
    if not ckpt:
        return None
    prov = ckpt.get("provenance") or []
    if not prov:
        # A format-1 checkpoint, or one written before provenance existed.
        # Nothing to compare against, so nothing honest to warn about.
        return None
    changed = execution_differs(prov[-1], profile)
    if not changed:
        return None
    lines = [
        "",
        "  " + "!" * 66,
        "  MIXED-EXECUTION LINEAGE",
        f"    checkpoint written by : {_describe_exec(prov[-1])}",
        f"    this run executes as  : {_describe_exec(profile)}",
        f"    changed               : {', '.join(changed)}",
        "",
        "    The weights continue and the objective is unchanged -- but the",
        "    epochs in this model were not all trained the same way. Report",
        "    that if you publish the run. Full history:",
    ]
    lines.extend(provenance_table(prov))
    lines.append("  " + "!" * 66)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  RANDOM STREAMS
# ════════════════════════════════════════════════════════════════════════════

def rng_state() -> Dict[str, object]:
    """
    Every random stream a Stage 10 training loop consumes.

    torch's is the one that matters: torch.multinomial inside
    _sample_candidates draws the K candidates, so without it a resumed run
    explores a different set of completions than the run it is continuing and
    the two are not the same experiment. Python's global stream is captured for
    completeness -- the loops do not currently draw from it, since the epoch
    order comes from a locally seeded Random -- so that adding any use of it
    later cannot silently break resume reproducibility.
    """
    import torch

    state: Dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Optional[Dict[str, object]], log=None) -> None:
    """
    Put the streams back.

    Missing or unusable state is a warning, not a failure: continuing with the
    right weights and a fresh RNG is far better than refusing to resume a
    ten-hour run. A checkpoint written by a stage that captured no RNG (or a
    format-1 one) simply passes None here.
    """
    if not state:
        return
    import torch

    try:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu()
                            if hasattr(state["torch"], "cpu") else state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:                             # pragma: no cover
        if log:
            log(f"  Could not restore RNG state ({e}); continuing with a fresh one.")


# ════════════════════════════════════════════════════════════════════════════
#  THE ROLLING CHECKPOINT
# ════════════════════════════════════════════════════════════════════════════

def save_state(
    save_dir:    str,
    stage:       str,
    *,
    trainable:   Mapping[str, object],
    optimizer:   Optional[dict],
    epoch:       int,
    batch_index: int,
    global_step: int,
    history:     Dict[str, list],
    agg:         Optional[Dict[str, float]] = None,
    n_steps:     int = 0,
    fingerprint: Optional[Dict[str, object]] = None,
    rng:         Optional[Dict[str, object]] = None,
    profile:     Optional[Dict[str, object]] = None,
    provenance:  Optional[List[dict]] = None,
    shared:      Optional[bool] = None,
) -> List[dict]:
    """
    Write the complete resumable state ATOMICALLY, and return the updated
    provenance list for the caller to keep in memory.

    `batch_index` is the index of the NEXT batch inside `epoch`, so a resume
    needs no "did the last one finish" reasoning: it starts there. A finished
    epoch is recorded as (epoch + 1, 0).

    The write goes to a temp file and is then os.replace'd over the target,
    which is atomic on POSIX and on Windows (MoveFileEx with REPLACE_EXISTING).
    The payload is ~93 MB and takes seconds on a mounted Drive, so an in-place
    write would leave a window in which a Ctrl-C or a pre-emption destroys the
    only checkpoint -- exactly the scenario this file exists to survive. The
    old state stays valid until the new one is complete on disk.
    """
    # Local import: this module is also used for path resolution alone, and
    # torch costs seconds to import.
    import torch

    _check_stage(stage)
    os.makedirs(save_dir, exist_ok=True)
    profile = profile or execution_profile(stage)
    # `epoch` is the NEXT epoch to run, which is the right thing for a resume
    # cursor and the wrong thing for a provenance label: a save at
    # (epoch=3, batch_index=0) means epoch 2 was just FINISHED, and a table
    # that said "epoch 3" for work done in epoch 2 would misattribute every
    # boundary save by one. batch_index == 0 is exactly the end-of-epoch case.
    trained_epoch = epoch - 1 if (batch_index == 0 and epoch > 1) else epoch
    prov = append_segment(provenance, profile, trained_epoch, global_step)

    payload = {
        "format":      CKPT_FORMAT,
        "epoch":       int(epoch),
        "batch_index": int(batch_index),
        "global_step": int(global_step),
        "history":     history,
        "agg":         agg or {},
        "n_steps":     int(n_steps),
        "fingerprint": fingerprint or {},
        "trainable":   trainable,
        "optimizer":   optimizer,
        "rng":         rng,
        "provenance":  prov,
        "written_by":  stage,
        "shared":      shared_enabled(shared, stage),
        "saved_at":    time.time(),
    }
    target = ckpt_path(save_dir, stage, shared)
    tmp = target + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, target)

    meta = {k: payload[k] for k in
            ("format", "epoch", "batch_index", "global_step", "n_steps",
             "fingerprint", "provenance", "written_by", "shared", "saved_at")}
    meta["history"] = history
    meta_target = meta_path(save_dir, stage, shared)
    tmp_meta = meta_target + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp_meta, meta_target)
    return prov


def load_state(save_dir: str, stage: str,
               shared: Optional[bool] = None,
               log=print) -> Optional[dict]:
    """
    The rolling checkpoint for this (save_dir, stage), or None.

    In shared mode this finds whatever the family last wrote, whichever stage
    wrote it. A checkpoint in Stage 10.1's format 1 loads unchanged -- it
    simply carries no provenance, which is reported rather than invented.

    A corrupt file reports and returns None rather than raising, so a damaged
    checkpoint costs the run its progress and not its ability to start.
    """
    import torch

    p = ckpt_path(save_dir, stage, shared)
    if not os.path.isfile(p):
        return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        if log:
            log(f"  Checkpoint at {p} could not be read ({e}); starting fresh.")
        return None
    ckpt.setdefault("provenance", [])
    ckpt.setdefault("written_by", None)
    return ckpt


def describe_mode(stage: str, variant: str) -> List[str]:
    """The lines a stage banner should print about where it writes and what it
    may inherit."""
    save_dir = resolve_save_dir(stage, variant)
    lines = [f"  output dir      : {save_dir}"]
    if shared_enabled(stage=stage):
        others = [s for s in LINEAGE_STAGES if s != stage]
        lines.append(f"  output mode     : SHARED with {', '.join(others)} "
                     f"[config.STAGE10_SHARED_OUTPUT=True]")
        lines.append(f"  checkpoint      : {LINEAGE_CKPT_NAME} -- a run started "
                     f"by any of the three continues here")
    else:
        lines.append("  output mode     : SEPARATE per stage "
                     "[config.STAGE10_SHARED_OUTPUT=False]")
    return lines


# ════════════════════════════════════════════════════════════════════════════
#  PROGRESS REPORTING
# ════════════════════════════════════════════════════════════════════════════

def _write(msg: str) -> None:
    """tqdm.write when tqdm is present -- it steps around a live bar -- else
    print. Both go to stdout, which is where every other banner in Stage 10
    goes; keeping the stream consistent is what makes a redirected log read in
    the right order."""
    if _tqdm is not None:
        _tqdm.write(msg)
    else:                                            # pragma: no cover
        print(msg)


def _fmt_secs(s: float) -> str:
    """Seconds as 1h23m / 12m34s / 45s -- short enough to sit inside a status
    line without pushing the numbers that matter off the end."""
    s = int(max(s, 0))
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


class Progress:
    """
    Progress reporting that suits the terminal it is actually writing to.

    On a TTY this is tqdm and nothing else: one live bar, redrawn in place on
    every step, exactly as these scripts have always shown it.

    When stderr is NOT a TTY -- Colab's `!python`, nohup, a piped log, a CI
    runner -- there is nothing to redraw in place. Every refresh is appended
    instead, so a 4,019-batch epoch leaves 4,019 near-identical bar lines and
    the messages that matter (the resume banner, the per-epoch summary, an
    interrupt notice) are buried in them. So the bar is switched off there and
    ONE status line is written every `every` updates, carrying the same postfix
    the bar would have shown plus a rate and an ETA -- roughly 40 lines per
    epoch instead of 4,019, at a cadence that still tells you the run is alive.

    The call surface is tqdm's -- set_postfix_str / update / close -- so the
    training loops read identically either way and can be diffed against each
    other.

    `every` is the status cadence in updates; 0 silences the periodic line
    without touching the bar. `disable` is forwarded to tqdm and also silences
    the lines, which is what DDP's non-main ranks want.
    """

    def __init__(self, total: int, desc: str, *, every: int = 100,
                 unit: str = "batch", disable: bool = False):
        self.total   = max(int(total), 0)
        self.desc    = desc
        self.every   = max(int(every or 0), 0)
        self.disable = bool(disable)
        self.tty     = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.n       = 0
        self._postfix = ""
        self._t0      = time.monotonic()

        if _tqdm is not None:
            self.bar = _tqdm(total=self.total, desc=desc, unit=unit,
                             dynamic_ncols=True,
                             disable=self.disable or not self.tty)
        else:                                        # pragma: no cover
            self.bar = None

        if self._lines_mode:
            _write(f"  {desc}: {self.total} {unit}(s) to run; "
                   f"no TTY, so progress prints every {self.every} instead of "
                   f"drawing a bar.")

    @property
    def _lines_mode(self) -> bool:
        """Periodic lines are for exactly one case: wanted, but no bar to draw."""
        return not self.disable and not self.tty and self.every > 0

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:
        self._postfix = s
        if self.bar is not None and self.tty and not self.disable:
            self.bar.set_postfix_str(s, refresh=refresh)

    def update(self, k: int = 1) -> None:
        self.n += k
        if self.bar is not None and self.tty and not self.disable:
            self.bar.update(k)
            return
        # == total, not >= : a miscounted total must not degrade into a line
        # per step for the rest of the run.
        if self._lines_mode and (self.n % self.every == 0 or self.n == self.total):
            _write(self._status())

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        if self._lines_mode:
            _write(f"  {self.desc}: {self.n}/{self.total} done in "
                   f"{_fmt_secs(time.monotonic() - self._t0)}.")

    def _status(self) -> str:
        elapsed = time.monotonic() - self._t0
        rate    = elapsed / max(self.n, 1)
        left    = max(self.total - self.n, 0)
        pct     = 100.0 * self.n / max(self.total, 1)
        return (f"  [{self.n}/{self.total}  {pct:4.1f}%]  {self._postfix}  "
                f"{rate:.2f}s/it  elapsed {_fmt_secs(elapsed)}  "
                f"eta {_fmt_secs(rate * left)}")

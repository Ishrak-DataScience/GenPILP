# -*- coding: utf-8 -*-
"""
stage9_1_scoring_worker.py
==========================
Process-pool worker for Stage 9.1's parallel RDKit scoring (speedup 7b).

Why this lives in its own module rather than inside the Stage 9.1 script
------------------------------------------------------------------------
Windows and Colab both start subprocesses with "spawn", not "fork": the child
is a fresh interpreter that re-imports whatever module the target function came
from and looks the function up by qualified name. A function defined in a
script's __main__ is therefore resolved by re-running that script through
runpy -- which works, but is fragile, and is doubly fragile when the script's
filename contains spaces and leads with a digit (as Stage 9.1's does, by
request). A plain importable module sidesteps that entirely: the child does
`import stage9_1_scoring_worker` and finds the function.

What the workers do and do NOT do
----------------------------------
They measure the RDKit terms only -- validity, QED, SA, novelty and the
PAINS/Brenk alerts -- via compute_property_components(need_tox21=False).

The Tox21 term is deliberately excluded. It is a torch classifier, so loading
it once per worker would multiply its memory by the pool size and have every
worker contend for the same GPU. Stage 9.1 scores it instead as a single
batched forward on the parent process (see stage9's score_tox21_batch) and
merges the two halves with compose_stage9_score, which is the one place the
weighted sum that defines the objective is written down.

n_alerts is skipped too (need_alert_count=False): it costs ~6.5 ms of the
~17 ms measurement and the score never reads it -- only the eval pass does.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def init_worker(blas_threads: int = 1) -> None:
    """
    Pool initializer: pin this worker's BLAS/OpenMP threads, then pay the
    one-off import and catalog-build costs once per worker instead of on
    whichever molecule happens to arrive first.

    THREAD PINNING FIRST, AND WHY THE IMPORTS ARE FUNCTION-LOCAL
    ------------------------------------------------------------
    W workers each starting T threads produce W*T threads on a W-core
    allocation, and the contention routinely makes a pooled run SLOWER than a
    serial one -- the exact failure hardware_autotune.worker_init exists to
    prevent, which this initializer previously did not do.

    OMP_NUM_THREADS is read by the OpenMP runtime when it is first
    initialised, which happens on `import torch`. Setting it afterwards is a
    no-op. That is why this module imports nothing at the top level: under
    "spawn" the child re-imports only this module (typing alone), so torch is
    still unimported when the env vars below are written, and they take.

    `blas_threads` comes from config.STAGE9_1_WORKER_BLAS_THREADS via
    ScoringPool. 1 is correct whenever the pool is sized to the allocation;
    raise it only if you have deliberately left cores idle.

    Building the PAINS and Brenk FilterCatalogs takes ~85 ms. Without this the
    first task on each worker carries that latency, which on a short batch is
    a visible stall rather than an amortised cost.
    """
    import os

    n = str(max(1, int(blas_threads)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = n

    from stage9_masked_property_finetune import _get_alert_catalog

    # After the import above, torch exists in this process; set its intra-op
    # count too, since that one IS honoured post-import (unlike OMP's).
    try:
        import torch

        torch.set_num_threads(int(n))
    except Exception:                                # pragma: no cover
        pass

    _get_alert_catalog("pains")
    _get_alert_catalog("brenk")


def score_one(args: Tuple[str, str]) -> Dict[str, Optional[float]]:
    """
    Measure one (generated, parent) pair. Returns the raw
    compute_property_components dict -- None preserved for "not measurable",
    so the parent process can tell it apart from a genuine 0.0 exactly as in
    the serial path. "tox21" is always None here; the parent fills it in.
    """
    generated, parent = args
    from stage9_masked_property_finetune import compute_property_components

    return compute_property_components(
        generated, parent, need_alert_count=False, need_tox21=False,
    )


def score_many(pairs: List[Tuple[str, str]]) -> List[Dict[str, Optional[float]]]:
    """Serial fallback with the identical contract, used when no pool is active."""
    return [score_one(p) for p in pairs]

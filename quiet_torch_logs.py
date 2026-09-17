# -*- coding: utf-8 -*-
"""
quiet_torch_logs.py
===================
Drops third-party deprecation chatter that torch logs during import, before it
reaches the run log.

The one it exists for, two lines at the top of every Stage 10 run on the
server:

    W0917 22:45:41.341000 ... torch/utils/_pytree.py:630] <enum
    'KernelPreference'> is an Enum subclass and is now natively supported by
    torch.compile as an opaque value type. Calling register_constant() on Enum
    subclasses is deprecated and will be an error in a future release.
    W0917 22:45:41.479000 ... <enum 'ScaleCalculationMode'> ...

Nothing in this project calls register_constant. torchao does, on its own
enums, while transformers imports it to see whether torchao quantisation is
available -- so the deprecation is a conversation between two libraries that
this pipeline only happens to be standing next to. It is not actionable here:
the fix belongs in torchao, and the only local alternatives are pinning torch
back or uninstalling torchao, both of which change the training environment to
silence a message that means nothing to this run.

WHY NOT A BLANKET FILTER. The message is dropped by SUBSTRING, so every other
warning torch emits still lands. A run that later hits a real torch
deprecation -- an AMP API, a DDP argument -- must still say so; those are the
ones that eventually break a training script, and they look exactly like this
one in the log.

Import it FIRST, above torch and transformers:

    import quiet_torch_logs  # noqa: F401   (silences torchao's deprecation)

Import order is the whole mechanism. torchao logs these while it is being
imported, so a filter installed afterwards has nothing left to catch.
"""

from __future__ import annotations

import logging
import warnings
from typing import Iterable, Sequence

# Substrings, matched against the formatted message. "register_constant" alone
# would be enough today; it is written out in full so that a future torch
# deprecation which merely MENTIONS register_constant in passing is not
# swallowed along with it.
_DROP = (
    "register_constant() on Enum subclasses is deprecated",
)

# Where the message comes from. A logger is silenced by attaching the filter to
# it directly rather than by filtering the root: torch installs its own handler
# on the "torch" logger lazily, and a filter on a parent logger does NOT see
# records that propagate up from a child, so the child is the only reliable
# attachment point. If a future torch moves the message, add the new module
# here -- the substring match keeps the blast radius to that one message.
_LOGGERS = (
    "torch.utils._pytree",
)


class _DropSubstrings(logging.Filter):
    """Rejects records whose formatted message contains any of `patterns`."""

    def __init__(self, patterns: Sequence[str]) -> None:
        super().__init__()
        self.patterns = tuple(patterns)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:                             # pragma: no cover
            # A record that cannot even be formatted is not ours to judge;
            # let it through and let the handler deal with it.
            return True
        return not any(p in message for p in self.patterns)


def silence(patterns: Iterable[str] = _DROP,
            loggers: Iterable[str] = _LOGGERS) -> None:
    """
    Install the filter. Safe to call more than once -- each logger carries at
    most one of these, so repeated imports (Stage 10.4 imports 10.2 imports
    10.1, all of which do this) do not stack filters.
    """
    patterns = tuple(patterns)
    for name in loggers:
        log = logging.getLogger(name)
        if any(isinstance(f, _DropSubstrings) and f.patterns == patterns
               for f in log.filters):
            continue
        log.addFilter(_DropSubstrings(patterns))
    # The same message on the warnings channel, for the torch builds that
    # raise it as a DeprecationWarning instead of logging it.
    warnings.filterwarnings(
        "ignore", message=".*register_constant.*", category=DeprecationWarning)


silence()

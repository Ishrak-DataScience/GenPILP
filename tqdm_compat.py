# -*- coding: utf-8 -*-
"""
tqdm_compat.py
==============
A real fallback for tqdm, imported only when tqdm itself is missing.

Why this module exists
----------------------
Every script in the pipeline guarded its import the same way:

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable=None, **kwargs):
            return iterable if iterable is not None else range(0)

which is not a fallback but a second failure mode. A plain function has no
``.write`` attribute, and this pipeline calls ``tqdm.write`` in more than a
hundred places -- so a machine without tqdm did not quietly lose its progress
bars, it died with AttributeError at the first log line, somewhere deep in a
training run rather than at import. The wrapper below implements the small
surface the pipeline actually uses, so "no tqdm installed" degrades to "no
progress bars" and nothing else.

What it implements, and nothing more
------------------------------------
``tqdm(iterable=None, **kwargs)`` for iteration and as a bar handle, with
``update`` / ``set_postfix_str`` / ``set_description`` / ``refresh`` /
``close`` and the context-manager protocol; and ``tqdm.write`` as a static
method, which is the one piece the old stub lacked. Every keyword real tqdm
accepts is swallowed, so a call site never has to know which one it got.

The bar itself is deliberately SILENT: without tqdm there is no in-place
redraw to be had, and a fallback that printed a line per update would recreate
the orphaned-progress-line problem that stage10_lineage.Progress exists to
avoid. Messages passed to ``write`` are the only thing that reaches the
terminal, which is exactly what a log wants.
"""

from __future__ import annotations

import sys


class tqdm:                                          # noqa: N801 (mirrors tqdm's own name)
    """A silent stand-in for tqdm.tqdm with the same call surface."""

    def __init__(self, iterable=None, total=None, desc=None, disable=False,
                 **kwargs):
        self.iterable = iterable
        self.desc     = desc
        self.disable  = bool(disable)
        self.n        = 0
        self.postfix  = ""
        if total is None:
            try:
                total = len(iterable)                # type: ignore[arg-type]
            except (TypeError, AttributeError):
                total = None
        self.total = total

    # ── iteration ────────────────────────────────────────────────────────
    def __iter__(self):
        if self.iterable is None:
            return
        for item in self.iterable:
            self.n += 1
            yield item

    def __len__(self):
        if self.total is not None:
            return int(self.total)
        return len(self.iterable) if self.iterable is not None else 0

    # ── bar handle ───────────────────────────────────────────────────────
    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        self.postfix = s

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:
        pairs = dict(ordered_dict or {}, **kwargs)
        self.postfix = ", ".join(f"{k}={v}" for k, v in pairs.items())

    def set_description(self, desc=None, refresh: bool = True) -> None:
        self.desc = desc

    def refresh(self, *args, **kwargs) -> None:
        pass

    def clear(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "tqdm":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── the method the old stub was missing ──────────────────────────────
    @staticmethod
    def write(s: str = "", file=None, end=None) -> None:
        """
        Print one message. Real tqdm clears the bar first and redraws it after;
        with no bar on screen there is nothing to clear, so this is print with
        tqdm's signature. Flushed because these are progress messages: a
        buffered one that appears after the run it was describing is worse than
        none.
        """
        stream = sys.stdout if file is None else file
        if end is None:
            print(s, file=stream)
        else:
            print(s, file=stream, end=end)
        try:
            stream.flush()
        except Exception:                            # pragma: no cover
            pass


# tqdm exposes these as module-level names too; a couple of scripts may reach
# for them, and an AttributeError here would be the very bug this file fixes.
trange = lambda *a, **k: tqdm(range(*a), **k)        # noqa: E731

# -*- coding: utf-8 -*-
"""
config.py  --  picks the right per-machine config, automatically.

This file is NOT the configuration. Every stage script does `import config`,
and this module decides WHICH config_<platform>.py that should resolve to on
the machine doing the import, loads it, and then *becomes* it. Nothing else in
the pipeline changes: `config.STAGE10_DIR` and friends still work exactly as
before, and a script mutating `config.X` at runtime (Stage 9.1 and Stage 10.2
both do, when a --speed preset overrides a knob) mutates the real module.

The point is that you no longer copy or rename anything. Upload every config
to every machine; each one picks its own.

How a config is chosen
----------------------
1. $GENPLIP_CONFIG, if set, wins outright. It accepts a platform name
   ("colab"), a module name ("config_colab"), a filename or a full path. This
   is the escape hatch for a machine where the automatic rule is wrong, and it
   is what the parent process hands down to its pool workers (see below).

2. Otherwise: every config_*.py beside this file that declares
   CONFIG_PLATFORM is a candidate, and the one whose BASE_DIR EXISTS as a
   directory on this machine wins. That is the whole trick -- a Colab config
   points at /content/drive/..., an HPC config at /home/<user>/..., and only
   one of those is ever real on a given box.

   CONFIG_PLATFORM is what keeps a legacy or scratch config_*.py lying around
   in the repo from being picked up by accident; a file without it is ignored.
   (If NO config declares one, every config_*.py is considered instead, so
   this still does something sensible in a half-migrated checkout.)

3. If several match -- two configs whose BASE_DIRs both exist -- the most
   specific one (longest BASE_DIR) wins and a warning names the alternatives,
   because "deeper path" is the stronger claim to being written for this
   machine's exact layout. Set GENPLIP_CONFIG to settle it for good.

4. If none match, that is an error and not a silent fallback: running the
   pipeline against someone else's paths would fail later, further away, and
   after wasting GPU time. The message lists every candidate and its BASE_DIR.

Adding a new machine
--------------------
Drop a config_<name>.py beside this file, give it CONFIG_PLATFORM and a
BASE_DIR that exists there, and it is picked up with no code change.

Process pools
-------------
Stage 9.1 / 10.1 / 10.2 score across a process pool, and a spawn/forkserver
child re-imports this module. The parent exports its choice into
$GENPLIP_CONFIG, so children take branch 1: they cannot re-probe and reach a
different answer, and they do not each reprint the banner.
"""

import ast
import glob
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SELF = os.path.basename(os.path.abspath(__file__))
_ENV_VAR = "GENPLIP_CONFIG"


def _candidate_paths():
    """Every config_*.py beside this file, this selector excluded."""
    return sorted(
        p for p in glob.glob(os.path.join(_HERE, "config_*.py"))
        if os.path.basename(p) != _SELF
    )


def _probe(path):
    """
    (CONFIG_PLATFORM, BASE_DIR) read WITHOUT executing the file.

    Parsing rather than importing matters: probing must not run the import-time
    side effects of configs meant for other machines, and on the machine that
    does win we still want exactly one module executed.
    """
    found = {}
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None, None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ("CONFIG_PLATFORM", "BASE_DIR"):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass          # computed, not a literal -- not probeable
    return found.get("CONFIG_PLATFORM"), found.get("BASE_DIR")


def _resolve_forced(raw):
    """
    $GENPLIP_CONFIG -> a path. Accepts a path, a filename, a module name or a
    CONFIG_PLATFORM label.
    """
    raw = raw.strip()
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw
    stem = os.path.splitext(os.path.basename(raw))[0]
    for path in _candidate_paths():
        name = os.path.splitext(os.path.basename(path))[0]
        if stem in (name, name.replace("config_", "")):
            return path
        platform, _ = _probe(path)
        if platform and stem == platform:
            return path
    known = ", ".join(os.path.basename(p) for p in _candidate_paths()) or "(none found)"
    raise ImportError(
        "${} is set to {!r}, but no config beside {} matches it.\n"
        "  Candidates: {}".format(_ENV_VAR, raw, _HERE, known)
    )


def _select():
    """(path, announce). announce is False when the choice was handed to us."""
    forced = os.environ.get(_ENV_VAR, "").strip()
    if forced:
        return _resolve_forced(forced), False

    probed = [(p,) + _probe(p) for p in _candidate_paths()]
    if not probed:
        raise ImportError(
            "No config_*.py found beside {} in {}.\n"
            "  Every machine needs one (config_colab.py, config_HPC_jupyter.py, ...)."
            .format(_SELF, _HERE)
        )

    marked = [t for t in probed if t[1]]
    pool = marked or probed                      # tolerate an unmarked checkout
    matches = [t for t in pool if t[2] and os.path.isdir(t[2])]

    if not matches:
        lines = "\n".join(
            "    {:28s} {:26s} BASE_DIR={!r} {}".format(
                os.path.basename(p),
                "[" + (plat or "no CONFIG_PLATFORM") + "]",
                base,
                "(missing)" if base else "(unreadable)",
            )
            for p, plat, base in probed
        )
        raise ImportError(
            "config.py could not tell which machine this is: no candidate's "
            "BASE_DIR exists here.\n" + lines + "\n"
            "  Fix by adding a config for this machine, or by forcing one:\n"
            "    export {0}=colab                    (bash)\n"
            "    $env:{0} = 'colab'                  (PowerShell)\n"
            "    os.environ['{0}'] = 'colab'         (notebook, before the first import)"
            .format(_ENV_VAR)
        )

    if len(matches) > 1:
        matches.sort(key=lambda t: len(t[2]), reverse=True)
        others = ", ".join("{} ({})".format(os.path.basename(p), base)
                           for p, _, base in matches[1:])
        print("  [config] WARNING: {} configs have an existing BASE_DIR here; taking "
              "the most specific one.\n"
              "           also matched: {}\n"
              "           set ${} to choose explicitly."
              .format(len(matches), others, _ENV_VAR), file=sys.stderr)
    return matches[0][0], True


def _load(path):
    """Execute the chosen config and return it, registered under its own name."""
    name = os.path.splitext(os.path.basename(path))[0]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module            # set before exec, so a self-import works
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_path, _announce = _select()
_chosen = _load(_path)

# Provenance, so a stage banner (or a confused future reader) can say which
# file it is actually running on.
_chosen.CONFIG_NAME = os.path.splitext(os.path.basename(_path))[0]
_chosen.CONFIG_PATH = _path
if not hasattr(_chosen, "CONFIG_PLATFORM"):
    _chosen.CONFIG_PLATFORM = _chosen.CONFIG_NAME.replace("config_", "")

# Hand the decision to pool workers and any subprocess, so they skip probing.
os.environ.setdefault(_ENV_VAR, _chosen.CONFIG_NAME)

if _announce:
    print("  [config] {} -> {}  (BASE_DIR={})".format(
        _chosen.CONFIG_PLATFORM, os.path.basename(_path),
        getattr(_chosen, "BASE_DIR", "?")))

# Become the chosen module. `import config` from here on returns the real
# thing, so `config is config_colab` and a runtime `config.X = ...` is seen by
# every other module. globals().update keeps this shim usable too, for anything
# holding a reference to it from mid-import.
globals().update({k: v for k, v in vars(_chosen).items() if not k.startswith("__")})
sys.modules[__name__] = _chosen

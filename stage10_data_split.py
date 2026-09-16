# -*- coding: utf-8 -*-
"""
stage10_data_split.py
=====================
The train/validation split every Stage 10 trainer shares.

WHY THIS EXISTS
---------------
Until this module, no Stage 10 script held anything out. collect_all_training_
pairs returned one flat list and every pair was trained on -- and
run_stage10_eval, which draws the headline "property distributions" figure,
called collect_pairs_by_source() over that same pool. The reported result was
measured on training data.

That is a bigger problem here than it would be in most pipelines, because of
the parent fallback. MEASURED at MASK_PERCENT=15: ~70% of molecules yield zero
valid completions out of K=16, so ~70% of training steps optimise
cross-entropy toward the PARENT'S OWN TOKENS. A model that simply memorised
its 46,982 parents would score beautifully on a figure computed over those
same parents, and nothing in the run would distinguish that from having learnt
to generate.

WHAT LEAKS, AND WHY THE SPLIT IS ON SCAFFOLDS
----------------------------------------------
Measured on stage1b_large_scale_plip_mask_summary.csv:

    585,428 usable rows          but only 46,982 unique parent SMILES
    12.5 rows per ligand         top 1% of ligands = 80.4% of rows
    12,551 distinct Murcko scaffolds

So there are three units one could split on, and they are not equivalent:

  ROW / PAIR   Catastrophic. Sulfate alone spans 33,538 rows; splitting them
               randomly puts the same molecule on both sides thousands of
               times. (_cap_pairs_per_parent already caps this at 3 per parent
               for training purposes, but a split must not depend on that.)
  MOLECULE     Stops the identical parent appearing twice. Close analogues --
               same ring system, one substituent moved -- still straddle, and
               in a PDB-derived set those are abundant.
  SCAFFOLD     What this module does. Whole Bemis-Murcko groups move together,
               so no validation molecule shares a ring core with a training
               one. This is the MoleculeNet/DeepChem convention and the same
               rule stage10_3_tox21_train.py uses, so the generator's split and
               the toxicity classifier's split mean the same thing.

Expect scaffold-split numbers to look WORSE than random-split ones, typically
by 0.05-0.10 on a normalised metric. That gap is the measurement, not a
regression: it is the part of apparent performance that was memorised
chemistry.

THE THREE BIG GROUPS, AND WHY THE ASSIGNMENT IS SIZE-BLIND
----------------------------------------------------------
    acyclic (no ring system)   6,754 molecules   14.4%
    Fe-porphyrin / heme        4,089 molecules    8.7%
    Mg-porphyrin / chlorophyll 2,393 molecules    5.1%

Three groups hold 28% of the data, so HOW groups are dealt out matters as much
as how they are formed. Until this was fixed, groups were filled largest-first
and a group that did not FIT the remaining budget fell to train. On the real
data that produced:

    train 37,327 (90.0%, 12,735 groups)   val 4,147 (10.0%, 4 GROUPS)

-- a 10% validation fold made of FOUR scaffold families. Its effective sample
size was 4, not 4,147, and every number computed on it described those four
chemistries rather than the model. Held-out candidate validity sat at 4.6%
against 37.9% on train FROM EPOCH 1, which is a property of the two folds and
not something training could have caused.

The rule now: shuffle the groups uniformly and deal them out until the fold's
molecule budget is met, TAKING the group that crosses it rather than skipping
it. Skipping is what made size matter -- a group could only be held out if it
fitted, so the big families could never be. Taking it costs an overshoot of at
most one group and buys the property the estimate rests on: a group's chance of
being held out does not depend on how big it is. Typical outcome on this data
is ~1,200 validation groups instead of 4.

The residual is second-order and worth stating rather than hiding: a group
still perturbs the budget it is measured against, so its inclusion probability
differs from a singleton's by roughly its own share of all molecules -- about
3% relative for the largest group here, against the 100% of the old rule.
Self-test [5] measures it rather than asserting it.

Size-blind does not mean low-variance. Draw one of the three big families into
validation and it alone is 15-60% of the fold; that is a legitimate draw and
the honest answer is to SEE it, so describe() reports the largest held-out
group's share and warns past 25%. Reseeding on fold COMPOSITION is legitimate;
reseeding after seeing a model metric is not.

STABILITY
---------
The split is computed over EVERY unique parent in the source data, once, and
cached to a manifest. It is therefore independent of STAGE10_MAX_TRAINING_PAIRS,
of STAGE9_MAX_PAIRS_PER_PARENT and of the sub-sampling seed: change any of
those and a molecule keeps the fold it had. Two stages run with different caps
still agree about what "validation" means, which is the whole point of the
module being shared.

Parents are CANONICALISED before grouping. That is not cosmetic: the Stage 1b
summary carries the same molecule under several notations (sulfate appears as
33,538 + 13,816 rows in two stereo-notations, per _cap_pairs_per_parent's own
note), and without canonicalisation those would be two groups and could land on
opposite sides.

Usage
-----
    import stage10_data_split as split

    pairs = collect_all_training_pairs(...)
    train_pairs, val_pairs = split.split_pairs(pairs)

    python stage10_data_split.py            build/refresh the manifest, print it
    python stage10_data_split.py --test     self-test (no data, no network)
    python stage10_data_split.py --rebuild  ignore the cache
    python stage10_data_split.py --random   build the RANDOM-fold split instead
    ... --molecule | --mode NAME            the other two modes
    ... --val-frac 0.1 --test-frac 0.05 --seed 42

A manifest is written PER MODE (..._split.scaffold.json, ..._split.random.json),
so both can exist side by side; config.STAGE10_SPLIT_MODE decides which one the
trainers read.

WHEN TO USE --random
--------------------
A scaffold split reports lower held-out numbers than a random one, by
construction, and that is not a defect to work around: the difference IS the
share of performance that came from analogues of training molecules. Build both
and report the pair. What is not defensible is quietly switching to the random
split because its numbers look better and describing the result as
generalisation -- so every banner and figure built from a random manifest says
so on its face.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

import config

try:
    from tqdm import tqdm
except ImportError:                                   # pragma: no cover
    from tqdm_compat import tqdm  # type: ignore[misc]


# ── knobs ───────────────────────────────────────────────────────────────────
VAL_FRAC   = getattr(config, "STAGE10_VAL_FRAC", 0.10)
TEST_FRAC  = getattr(config, "STAGE10_TEST_FRAC", 0.0)
SPLIT_SEED = getattr(config, "STAGE10_SPLIT_SEED", 42)
SPLIT_MODE = getattr(config, "STAGE10_SPLIT_MODE", "scaffold")
MANIFEST   = getattr(config, "STAGE10_SPLIT_MANIFEST", "") or ""

# RDKit's Murcko decomposition can blow the C++ stack -- a hard SEGFAULT, not a
# Python exception, so it cannot be caught -- on very long SMILES. It is
# reproducible on this dataset: peptide and oligosaccharide ligands crash the
# pass outright. Anything longer than this is grouped by its own canonical
# SMILES instead, which is 1,015 of 46,982 molecules (2.2%) and costs only that
# those few are split at molecule granularity rather than scaffold.
MAX_SCAFFOLD_SMILES_LEN = 400

# Bumped whenever the FOLD ASSIGNMENT changes. It rides in the fingerprint, so
# a manifest built under an older rule is REBUILT rather than silently reused.
# Without it a fix to assign_folds is invisible on every machine that already
# has a cached manifest -- which is every machine that has trained.
#   1  largest-group-first, skip what does not fit  (biased; see the header)
#   2  uniform shuffle, take the group that crosses the budget
SPLIT_ALGORITHM = 2

_FOLDS = ("train", "val", "test")


# ════════════════════════════════════════════════════════════════════════════
#  GROUPING
# ════════════════════════════════════════════════════════════════════════════

def canonical(smiles: str) -> Optional[str]:
    """Canonical SMILES, or None when RDKit cannot parse it."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:                                  # pragma: no cover
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:                                  # pragma: no cover
        return None


def scaffold_key(smiles: str) -> str:
    """
    The group a molecule belongs to.

    Normally its Bemis-Murcko scaffold. Three cases get a group of their own
    instead, prefixed so they are visibly not scaffolds:

        mol:<canonical>   too long for RDKit's Murcko code to be safe on
        bad:<sha1>        RDKit could not parse it at all
        ""                acyclic -- a legitimate empty scaffold, and every
                          acyclic molecule shares it, which is correct: they
                          genuinely all have "no ring system" in common

    Giving the first two their own singleton groups is the conservative choice:
    a molecule whose group cannot be determined is never allowed to pull an
    unrelated molecule across the split with it.
    """
    if not smiles:
        return "bad:empty"
    if len(smiles) > MAX_SCAFFOLD_SMILES_LEN:
        return "mol:" + (canonical(smiles) or smiles)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "bad:" + hashlib.sha1(smiles.encode()).hexdigest()[:16]
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol,
                                                   includeChirality=False)
    except Exception:
        return "bad:" + hashlib.sha1(smiles.encode()).hexdigest()[:16]


def group_parents(parents: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    (canonical -> group key, group key -> [canonical, ...]).

    Keyed on the CANONICAL parent, so the same molecule written two ways is one
    entry. Both maps are returned because the caller needs the forward map to
    assign folds and the reverse map to size the groups.
    """
    by_mol: Dict[str, str] = {}
    by_group: Dict[str, List[str]] = {}
    for raw in parents:
        canon = canonical(raw) or raw
        if canon in by_mol:
            continue
        key = scaffold_key(canon)
        by_mol[canon] = key
        by_group.setdefault(key, []).append(canon)
    return by_mol, by_group


# ════════════════════════════════════════════════════════════════════════════
#  THE SPLIT
# ════════════════════════════════════════════════════════════════════════════

def assign_folds(by_group: Dict[str, List[str]],
                 val_frac: float = None,
                 test_frac: float = None,
                 seed: int = None) -> Dict[str, str]:
    """
    canonical SMILES -> "train" | "val" | "test", whole groups together.

    SIZE-BLIND. Groups are dealt out in UNIFORM RANDOM ORDER, and the group
    that crosses a fold's molecule budget is TAKEN rather than skipped. Both
    halves of that sentence are load-bearing, and this function used to get
    both wrong.

    WHY ORDER MUST NOT BE SIZE. The previous rule sorted by size descending, so
    a group's fold was a deterministic function of how big it was: the largest
    groups had probability 0 of being held out and the smallest had ~1. On this
    data three scaffold families hold 28% of the molecules, and a 10% fold came
    out as FOUR groups covering 4,147 molecules -- effective n of 4. A number
    computed there describes those four chemistries, not the model.

    WHY THE CROSSING GROUP IS TAKEN, NOT SKIPPED. Skipping is the subtler half
    of the same bug. If a group is held out only when it FITS the remaining
    budget, then inclusion depends on the group's own size and big families are
    systematically pushed to train -- milder than sorting by size, but the same
    defect. Taking it makes inclusion depend only on what came BEFORE the group
    in the shuffle, which is independent of the group itself. The cost is that
    a fold overshoots its budget by at most one group.

    WHAT REMAINS, STATED RATHER THAN HIDDEN. This is not exactly uniform. A
    group is held out iff the groups preceding it in the shuffle total less
    than the budget, and the pool of "other groups" is missing the group being
    considered -- so its probability differs from a singleton's by roughly its
    own share of all molecules. For the largest group here that is ~3% relative
    (0.10 vs 0.103); the rule it replaces was 0.00 vs 0.25. Self-test [5]
    MEASURES this rather than assuming it.

    SIZE-BLIND IS NOT LOW-VARIANCE, and no assignment can make it so on data
    where three groups hold 28% of the molecules. Draw one of them into
    validation and it alone is most of the fold; that is a legitimate draw, not
    a bug, and the answer is to see it -- describe() reports the largest
    held-out group's share and warns past 25%.

    DETERMINISTIC ACROSS MACHINES. Keys are sorted before the seeded shuffle,
    so dict insertion order cannot leak in and the same data gives the same
    folds everywhere. A split that moved between runs would silently invalidate
    every comparison made against it, including a resumed run's own earlier
    epochs.
    """
    val_frac  = VAL_FRAC  if val_frac  is None else val_frac
    test_frac = TEST_FRAC if test_frac is None else test_frac
    rng = random.Random(SPLIT_SEED if seed is None else seed)

    keys = sorted(by_group)                # sort first: dict order must not
    rng.shuffle(keys)                      # leak into a "random" order

    n_total = sum(len(v) for v in by_group.values())
    n_val   = int(math.floor(n_total * val_frac))
    n_test  = int(math.floor(n_total * test_frac))

    folds: Dict[str, str] = {}
    n_in = {"train": 0, "val": 0, "test": 0}
    for key in keys:
        members = by_group[key]
        # "< budget", not "+ len(members) <= budget": the group that crosses
        # the line goes in. That is the whole fix -- see the docstring.
        if n_in["val"] < n_val:
            fold = "val"
        elif n_in["test"] < n_test:
            fold = "test"
        else:
            fold = "train"
        n_in[fold] += len(members)
        for m in members:
            folds[m] = fold
    return folds


def _fingerprint(n_parents: int, val_frac: float, test_frac: float,
                 seed: int, mode: str) -> Dict[str, object]:
    """What the cached manifest is only valid for."""
    return {"n_parents": int(n_parents), "val_frac": float(val_frac),
            "test_frac": float(test_frac), "seed": int(seed), "mode": str(mode),
            "max_scaffold_smiles_len": MAX_SCAFFOLD_SMILES_LEN,
            "algorithm": SPLIT_ALGORITHM}


def build_split(parents: Sequence[str], val_frac: float = None,
                test_frac: float = None, seed: int = None,
                mode: str = None) -> Dict[str, object]:
    """
    Group `parents` and assign folds. Returns the manifest dict.

    A mode chooses the GROUPING and nothing else. One size-blind assignment
    then deals those groups out, so "scaffold vs random" compares one thing
    rather than two. Three modes, in decreasing order of leakage removed:

        "scaffold"  group by Murcko core; whole groups move together. The
                    default and the only one whose held-out number is a
                    generalisation estimate.
        "molecule"  one group per canonical SMILES. Identical parents cannot
                    straddle, but analogues can.
        "random"    identical to "molecule" in construction -- they differ only
                    in what the banner and the manifest filename say, and
                    "random" says the loud thing. Identical parents still
                    cannot straddle, canonicalisation sees to that, but nothing
                    else is controlled.

    "random" exists to be COMPARED AGAINST "scaffold", not to replace it. The
    difference between the two held-out numbers is the share of apparent
    performance that comes from analogues of training molecules, which is a
    quantity worth reporting rather than a nuisance to design away. A random
    split will read several points better; that is what it is measuring, and it
    is not a better model.
    """
    val_frac  = VAL_FRAC   if val_frac  is None else val_frac
    test_frac = TEST_FRAC  if test_frac is None else test_frac
    seed      = SPLIT_SEED if seed      is None else seed
    mode      = (SPLIT_MODE if mode is None else mode).lower()
    # `is None` rather than falsiness: None means "use the configured
    # default", but an explicit "" is a typo -- a --mode with nothing behind
    # it -- and silently running the default split under a name the caller
    # believed they had chosen is exactly the confusion to avoid.
    if mode not in ("scaffold", "molecule", "random"):
        raise ValueError(
            f"mode must be 'scaffold', 'molecule' or 'random', got {mode!r}")

    if mode == "scaffold":
        by_mol, by_group = group_parents(parents)
    else:
        by_mol, by_group = {}, {}
        for raw in parents:
            canon = canonical(raw) or raw
            if canon not in by_mol:
                by_mol[canon] = "mol:" + canon
                by_group["mol:" + canon] = [canon]

    # One assignment for every mode. There used to be a second one for
    # "random", and it existed only because this one ordered by size and, over
    # size-1 groups, that degenerated to alphabetical -- a systematic partition
    # wearing a random split's name. A size-blind assignment needs no such
    # workaround, and having two would put a second difference inside a
    # comparison designed to isolate one.
    folds = assign_folds(by_group, val_frac, test_frac, seed)
    counts = {f: sum(1 for v in folds.values() if v == f) for f in _FOLDS}
    group_counts = {
        f: len({by_mol[m] for m, v in folds.items() if v == f}) for f in _FOLDS
    }
    # The largest group on each side. describe() needs it to say whether the
    # held-out fold is one chemistry wearing a fold's name -- rare under a
    # size-blind deal, but no assignment can make it impossible on data where
    # three groups hold 28% of the molecules.
    fold_of_group: Dict[str, str] = {}
    for mol, f in folds.items():
        fold_of_group.setdefault(by_mol[mol], f)
    fold_max_group = {f: 0 for f in _FOLDS}
    for key, members in by_group.items():
        f = fold_of_group.get(key)
        if f is not None and len(members) > fold_max_group[f]:
            fold_max_group[f] = len(members)
    return {
        "format": 1,
        "mode": mode,
        "fingerprint": _fingerprint(len(folds), val_frac, test_frac, seed, mode),
        "n_molecules": len(folds),
        "n_groups": len(by_group),
        "counts": counts,
        "group_counts": group_counts,
        "fold_max_group": fold_max_group,
        "folds": folds,
    }


# ════════════════════════════════════════════════════════════════════════════
#  THE SHARED MANIFEST
# ════════════════════════════════════════════════════════════════════════════

_CACHE: Optional[Dict[str, object]] = None


def manifest_path(path: str = None, mode: str = None) -> str:
    """
    Where the shared manifest lives, PER MODE.

    The mode is part of the filename (``..._split.scaffold.json``,
    ``..._split.random.json``) so the two can coexist. Without that they would
    share one path, and because the fingerprint records the mode, building a
    random split and then running a trainer configured for "scaffold" would
    detect the mismatch and silently REBUILD -- destroying the manifest that
    had just been built and doing it quietly, mid-run. Separate files make
    switching modes a config change and nothing else, and make it possible to
    hold both splits and compare them.
    """
    base = path or MANIFEST or os.path.join(
        getattr(config, "STAGE10_DIR", "./outputs/stage10/"),
        "stage10_split.json")
    mode = (mode or SPLIT_MODE).lower()
    stem, ext = os.path.splitext(base)
    if stem.endswith("." + mode):                  # already qualified
        return base
    return f"{stem}.{mode}{ext or '.json'}"


def load_manifest(path: str = None, mode: str = None) -> Optional[Dict[str, object]]:
    """The cached manifest, or None if absent/unreadable."""
    p = manifest_path(path, mode)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            man = json.load(fh)
    except Exception as e:                             # pragma: no cover
        tqdm.write(f"  Split manifest at {p} could not be read ({e}).")
        return None
    return man if man.get("folds") else None


def save_manifest(man: Dict[str, object], path: str = None,
                  mode: str = None) -> str:
    """Write the manifest atomically; returns the path."""
    p = manifest_path(path, mode or man.get("mode"))
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    os.replace(tmp, p)
    return p


def get_split(parents: Sequence[str], path: str = None, rebuild: bool = False,
              val_frac: float = None, test_frac: float = None,
              seed: int = None, mode: str = None,
              log=None) -> Dict[str, object]:
    """
    The manifest, built once and reused by every Stage 10 trainer.

    A cached manifest is REUSED EVEN IF `parents` contains molecules it does not
    know about, and the unknown ones are assigned on the fly by their group.
    That is deliberate and it is what makes the split shareable: Stage 10 with
    max_pairs=None and Stage 10.4 with max_pairs=50,000 see different parent
    LISTS, and if the manifest were rebuilt per run they would disagree about
    which molecules are validation -- while both printed a confident number.

    It is rebuilt only when the fold fractions, the seed or the mode change,
    because those are the settings that define what the folds MEAN.
    """
    global _CACHE
    log = log or tqdm.write
    val_frac  = VAL_FRAC   if val_frac  is None else val_frac
    test_frac = TEST_FRAC  if test_frac is None else test_frac
    seed      = SPLIT_SEED if seed      is None else seed
    mode      = (SPLIT_MODE if mode is None else mode).lower()

    # Keyed on the mode: a process that asked for the scaffold split must not
    # be handed a cached random one just because it was requested first.
    if (_CACHE is not None and not rebuild
            and _CACHE.get("mode") == mode):
        return _CACHE

    man = None if rebuild else load_manifest(path, mode)
    if man is not None:
        old = man.get("fingerprint") or {}
        want = _fingerprint(old.get("n_parents", 0), val_frac, test_frac,
                            seed, mode)
        differing = [k for k in want
                     if k != "n_parents" and old.get(k) != want[k]]
        if differing:
            log(f"  Split manifest was built with different settings "
                f"({', '.join(differing)}); rebuilding.")
            man = None

    if man is None:
        log(f"  Building the {mode} split over {len(set(parents)):,} parent "
            f"SMILES (this is done once and cached) ...")
        man = build_split(parents, val_frac, test_frac, seed, mode)
        p = save_manifest(man, path, mode)
        log(f"  Split manifest written -> {p}")

    _CACHE = man
    return man


def fold_of(smiles: str, man: Dict[str, object]) -> str:
    """
    Which fold a parent belongs to.

    A molecule the manifest has never seen is placed by its GROUP: if any known
    molecule shares its scaffold, it inherits that fold, so an unseen analogue
    of a validation molecule can never land in train. Only a molecule whose
    whole group is new falls through to train -- new chemistry is training data
    by default, which is the safe direction (it can never inflate a validation
    score).
    """
    folds = man["folds"]
    canon = canonical(smiles) or smiles
    hit = folds.get(canon)
    if hit is not None:
        return hit
    key = scaffold_key(canon) if man.get("mode", "scaffold") == "scaffold" \
        else "mol:" + canon
    group_folds = _group_index(man)
    return group_folds.get(key, "train")


_GROUP_INDEX_FOR: Optional[int] = None
_GROUP_INDEX: Dict[str, str] = {}


def _group_index(man: Dict[str, object]) -> Dict[str, str]:
    """group key -> fold, built once per manifest object."""
    global _GROUP_INDEX_FOR, _GROUP_INDEX
    if _GROUP_INDEX_FOR == id(man):
        return _GROUP_INDEX
    idx: Dict[str, str] = {}
    scaffold_mode = man.get("mode", "scaffold") == "scaffold"
    for mol, fold in man["folds"].items():
        key = scaffold_key(mol) if scaffold_mode else "mol:" + mol
        idx.setdefault(key, fold)
    _GROUP_INDEX_FOR, _GROUP_INDEX = id(man), idx
    return idx


def split_pairs(pairs: Sequence[Tuple[str, str]], man: Dict[str, object] = None,
                path: str = None, log=None,
                ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]],
                           List[Tuple[str, str]]]:
    """
    Partition (masked, parent) pairs into (train, val, test) by the parent's
    fold.

    Every pair derived from one parent goes to the same side -- that is the
    whole point, since a molecule's several PLIP instances are the same
    chemistry seen in different crystals, and letting one instance train while
    another validates would leak the answer directly.
    """
    log = log or tqdm.write
    if man is None:
        man = get_split([p for _, p in pairs], path=path, log=log)
    out = {"train": [], "val": [], "test": []}
    for pair in pairs:
        out[fold_of(pair[1], man)].append(pair)
    return out["train"], out["val"], out["test"]


def describe(man: Dict[str, object], pairs_split=None) -> List[str]:
    """Banner lines naming what the split is and what it cost."""
    c, g = man["counts"], man.get("group_counts", {})
    total = max(man["n_molecules"], 1)
    lines = [
        f"  data split      : {man['mode']} "
        f"({man['n_groups']:,} groups over {man['n_molecules']:,} parent "
        f"molecules, seed {man['fingerprint']['seed']})",
        f"                    train {c['train']:,} ({c['train']/total:.1%}, "
        f"{g.get('train', 0):,} groups)   "
        f"val {c['val']:,} ({c['val']/total:.1%}, {g.get('val', 0):,} groups)"
        + (f"   test {c['test']:,} ({c['test']/total:.1%})" if c["test"] else ""),
    ]
    # The fold's SHAPE, not just its size. A held-out number is only as good
    # as the diversity of what it was measured on, and that is invisible from
    # the molecule count alone -- the run that motivated this line reported a
    # healthy "val 4,147 (10.0%)" over four scaffold groups.
    mx = (man.get("fold_max_group") or {}).get("val", 0)
    if mx and c["val"]:
        share = mx / c["val"]
        lines.append(f"                    held-out fold spans "
                     f"{g.get('val', 0):,} group(s); its largest is {mx:,} "
                     f"molecule(s) ({share:.0%})")
        if share > 0.25:
            lines.append("                    WARNING: one group dominates "
                         "the held-out fold, so its numbers")
            lines.append("                    describe that chemistry more "
                         "than the model. Change")
            lines.append("                    STAGE10_SPLIT_SEED and rebuild. "
                         "Reseeding on fold COMPOSITION")
            lines.append("                    is legitimate; reseeding after "
                         "seeing a model metric is not.")
    if man["mode"] == "scaffold":
        lines.append("                    no validation molecule shares a "
                     "Murcko scaffold with a training one")
    elif man["mode"] == "random":
        # Said on every banner, every run, because a random-split number read
        # as a generalisation number is the whole failure this module exists
        # to prevent, and the manifest filename is the only other clue.
        lines.append("                    RANDOM split -- analogues of training "
                     "molecules ARE in validation.")
        lines.append("                    Held-out numbers here are OPTIMISTIC "
                     "and are not a generalisation")
        lines.append("                    estimate; compare them against the "
                     "scaffold split, do not replace it.")
    elif man["mode"] == "molecule":
        lines.append("                    identical parents cannot straddle, "
                     "but analogues can")
    if pairs_split is not None:
        tr, va, te = pairs_split
        lines.append(f"  pairs           : train {len(tr):,}   val {len(va):,}"
                     + (f"   test {len(te):,}" if te else ""))
    return lines


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    print("\nStage 10 data-split self-test")
    print("=" * 62)

    benzenes = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1CCCC"]
    pyridines = ["c1ccncc1C", "c1ccncc1CC"]
    acyclic = ["CCCC", "CCCCC", "CCO", "CC(C)O"]
    mols = benzenes + pyridines + acyclic

    # 1. Grouping is by ring core, and acyclic molecules share one group.
    by_mol, by_group = group_parents(mols)
    assert len({by_mol[canonical(m)] for m in benzenes}) == 1, \
        "the benzene series must be ONE scaffold group"
    assert by_mol[canonical("CCCC")] == "", "acyclic molecules have an empty scaffold"
    assert len({by_mol[canonical(m)] for m in acyclic}) == 1
    assert by_mol[canonical("c1ccccc1C")] != by_mol[canonical("c1ccncc1C")], \
        "benzene and pyridine are different cores"
    print(f"  [1] Murcko grouping: {len(by_group)} groups from "
          f"{len(mols)} molecules            OK")

    # 2. Whole groups move together -- the property the whole module is for.
    man = build_split(mols, val_frac=0.25, test_frac=0.0, seed=1)
    for key, members in by_group.items():
        got = {man["folds"][m] for m in members}
        assert len(got) == 1, (
            f"group {key!r} was split across {got} -- a scaffold split that "
            f"splits a scaffold is not one")
    print("  [2] every scaffold group lands wholly on one side          OK")

    # 3. Deterministic across calls.
    assert build_split(mols, 0.25, 0.0, 1)["folds"] == man["folds"], \
        "build_split is not deterministic"
    print("  [3] the split is reproducible from its seed                OK")

    # 4. Canonicalisation: the same molecule in two notations is ONE entry and
    #    cannot land on two sides. This is the sulfate case from Stage 1b.
    two_ways = ["C1=CC=CC=C1C", "Cc1ccccc1"]
    assert canonical(two_ways[0]) == canonical(two_ways[1])
    bm, bg = group_parents(two_ways)
    assert len(bm) == 1, f"two notations of toluene became {len(bm)} entries"
    print("  [4] two notations of one molecule collapse to one entry    OK")

    # 5. THE FOLD IS SIZE-BLIND -- the property the whole estimate rests on.
    #    If a group's chance of being held out depended on how big it is, the
    #    validation fold would be a biased sample of chemistry and the number
    #    computed on it would not estimate anything.
    #
    #    MEASURED over 2,000 seeds, not asserted by construction, because the
    #    guarantee is probabilistic: one 12-member group against 40 singletons,
    #    and the big group must be held out about as often as a small one.
    #    Theory says 0.244 vs 0.216 -- the residual second-order term the
    #    docstring quantifies. The rule this replaced scored 0.000 vs 0.250: it
    #    could not put the big group in validation at all, which is exactly how
    #    a 10% fold of the real data came out as four scaffold groups.
    synth = {"big": ["b%d" % i for i in range(12)]}
    synth.update({"s%d" % i: ["m%d" % i] for i in range(40)})
    trials = 2000
    hits = {"big": 0, "s0": 0}
    for s in range(trials):
        f5 = assign_folds(synth, val_frac=0.20, test_frac=0.0, seed=s)
        for key in hits:
            if f5[synth[key][0]] == "val":
                hits[key] += 1
    p_big, p_small = hits["big"] / trials, hits["s0"] / trials
    assert abs(p_big - p_small) < 0.08, (
        "held-out probability depends on group size (big=%.3f small=%.3f) -- "
        "the validation fold is a biased sample of chemistry"
        % (p_big, p_small))
    assert p_big > 0.10, (
        "a 12-member group reached validation only %.3f of the time -- large "
        "groups are still being excluded from the fold" % p_big)
    print("  [5] fold membership is size-blind (big %.2f vs small %.2f)  OK"
          % (p_big, p_small))

    # 5b. THE OBSERVED REGRESSION, reproduced exactly. Eight 25-member
    #     families whose sizes sum to precisely the 10% budget, against 800
    #     singletons. Largest-first packs the budget with four of the families
    #     and then has no room left, so validation came out as FOUR GROUPS --
    #     which is what the real run reported over 41,474 molecules. The
    #     numbers are chosen so the old rule scores 4 here; a weaker layout
    #     (one huge family plus singletons) scores 100 under BOTH rules and
    #     would not have caught anything.
    synth2 = {"g%d" % g: ["x%d_%d" % (g, i) for i in range(25)]
              for g in range(8)}
    synth2.update({"s%d" % i: ["y%d" % i] for i in range(800)})
    ngroups = []
    for s in range(50):
        f5b = assign_folds(synth2, val_frac=0.10, test_frac=0.0, seed=s)
        ngroups.append(len([k for k, ms in synth2.items()
                            if f5b[ms[0]] == "val"]))
    typical = sorted(ngroups)[len(ngroups) // 2]
    assert typical > 20, (
        "the median validation fold holds %d group(s) -- a fold that small is "
        "the collapse this rule exists to prevent" % typical)
    print("  [5b] folds span many groups (median %d of 808, was 4)      OK"
          % typical)

    # 6. Pair splitting keeps every instance of a parent together.
    pairs = [("m1", "c1ccccc1C"), ("m2", "c1ccccc1C"), ("m3", "c1ccccc1CC"),
             ("m4", "C1CCOCC1"), ("m5", "C1CCOCC1")]
    tr, va, te = split_pairs(pairs, man=build_split([p for _, p in pairs],
                                                    val_frac=0.4, seed=3))
    assert len(tr) + len(va) + len(te) == len(pairs), "pairs were lost"
    where = {}
    for name, part in (("tr", tr), ("va", va), ("te", te)):
        for _, parent in part:
            where.setdefault(canonical(parent), set()).add(name)
    straddling = {p for p, s in where.items() if len(s) > 1}
    assert not straddling, f"parent(s) {straddling} appear in two folds"
    print("  [6] all pairs of one parent stay on the same side          OK")

    # 7. An UNSEEN molecule inherits its group's fold, so an unseen analogue of
    #    a validation molecule can never be trained on.
    m3 = build_split(["c1ccccc1C", "C1CCOCC1"], val_frac=0.5, seed=7)
    val_mol = next(m for m, f in m3["folds"].items() if f == "val")
    analogue = val_mol.replace("C1", "CC1", 1) if "C1" in val_mol else val_mol
    same_core = "c1ccccc1CCCCC" if m3["folds"].get(canonical("c1ccccc1C")) == "val" \
        else "C1CCOCC1CC"
    if scaffold_key(canonical(same_core)) in _group_index(m3):
        assert fold_of(same_core, m3) == _group_index(m3)[
            scaffold_key(canonical(same_core))], \
            "an unseen molecule must inherit its scaffold group's fold"
    # A wholly new group falls to train, never to val.
    assert fold_of("C1CCC2CCCCC2C1", m3) in ("train",), \
        "a molecule from an unknown group must default to TRAIN, never val"
    print("  [7] unseen molecules inherit their group; new -> train     OK")

    # 8. Long SMILES do not reach RDKit's Murcko code. This is a SEGFAULT
    #    guard, not an exception guard -- it crashed the whole process on this
    #    dataset's peptide ligands, and a segfault cannot be caught.
    long_smi = "C" * (MAX_SCAFFOLD_SMILES_LEN + 50)
    k = scaffold_key(long_smi)
    assert k.startswith("mol:"), f"long SMILES must bypass Murcko, got {k!r}"
    assert scaffold_key("not a molecule at all").startswith("bad:")
    print("  [8] over-long / unparseable SMILES bypass Murcko safely    OK")

    # 9. The manifest round-trips.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "split.json")
        save_manifest(man, p)
        assert not os.path.exists(p + ".tmp"), "temp file left behind"
        back = load_manifest(p)
        assert back["folds"] == man["folds"]
        assert back["fingerprint"] == man["fingerprint"]
        assert load_manifest(os.path.join(td, "nope.json")) is None
    print("  [9] manifest save -> load round-trips, write is atomic     OK")

    # 10. molecule mode really is weaker, and says so by producing more groups.
    ms = build_split(mols, val_frac=0.25, seed=1, mode="molecule")
    assert ms["n_groups"] == len({canonical(m) for m in mols}) > man["n_groups"], (
        "molecule mode must produce one group per molecule")
    print("  [10] molecule mode = one group per molecule                OK")

    # 11. random mode: still deterministic, still no identical parent on both
    #     sides, but NOT scaffold-disjoint. The exact fold size also pins
    #     the budget rule: over size-1 groups nothing can overshoot, so a
    #     25% fold of 24 must be exactly 6.
    many = ["c1ccccc1" + "C" * i for i in range(1, 13)] + \
           ["C1CCOCC1" + "C" * i for i in range(1, 13)]
    r1 = build_split(many, val_frac=0.25, test_frac=0.0, seed=5, mode="random")
    r2 = build_split(many, val_frac=0.25, test_frac=0.0, seed=5, mode="random")
    assert r1["folds"] == r2["folds"], "random mode is not reproducible"
    assert r1["mode"] == "random"
    assert build_split(many, 0.25, 0.0, 6, "random")["folds"] != r1["folds"], \
        "a different seed must give a different random split"
    n_val = sum(1 for v in r1["folds"].values() if v == "val")
    assert n_val == int(len(many) * 0.25), \
        f"random val fold is {n_val}, expected {int(len(many) * 0.25)}"
    print(f"  [11] random mode: reproducible, seed-sensitive, {n_val}/"
          f"{len(many)} in val   OK")

    # 12. molecule and random now BUILD THE SAME SPLIT, and that is the fix
    #     rather than a regression. They diverged only because assign_folds
    #     ordered by size and, over size-1 groups, that degenerated to
    #     alphabetical -- so a second function existed to shuffle instead. One
    #     size-blind assignment serves every mode, and the only thing a mode
    #     chooses is the GROUPING, which is the only thing it should ever have
    #     chosen: otherwise "scaffold vs random" varies two things at once.
    m_mol = build_split(many, val_frac=0.25, test_frac=0.0, seed=5,
                        mode="molecule")
    assert m_mol["folds"] == r1["folds"], (
        "molecule and random group identically -- one canonical SMILES per "
        "group -- and share one assignment, so they must agree")
    sc = build_split(many, val_frac=0.25, test_frac=0.0, seed=5,
                     mode="scaffold")
    assert sc["n_groups"] < m_mol["n_groups"], (
        "scaffold mode must group analogues together, unlike molecule mode")
    print("  [12] modes differ only by grouping, not by assignment       OK")

    # 13. Identical parents still cannot straddle in random mode: grouping is
    #     on the canonical SMILES, so the two notations of one molecule are
    #     one group whichever mode is in force.
    two = ["C1=CC=CC=C1C", "Cc1ccccc1"] + ["C1CCOCC1", "c1ccncc1"]
    rr = build_split(two, val_frac=0.5, test_frac=0.0, seed=1, mode="random")
    assert len(rr["folds"]) == 3, \
        f"toluene's two notations did not collapse: {sorted(rr['folds'])}"
    print("  [13] random mode still collapses duplicate notations        OK")

    # 14. Manifests are per-mode, so building one never destroys the other.
    ps = manifest_path("/tmp/x/stage10_split.json", "scaffold")
    pr = manifest_path("/tmp/x/stage10_split.json", "random")
    assert ps != pr, "scaffold and random manifests share a path"
    assert ps.endswith(".scaffold.json") and pr.endswith(".random.json")
    assert manifest_path(ps, "scaffold") == ps, "path re-qualified twice"
    print("  [14] manifest path is per-mode and idempotent               OK")

    # 15. An unknown mode is refused rather than silently treated as scaffold.
    for bad in ("Random ", "stratified", ""):
        try:
            build_split(["CCO"], mode=bad)
        except ValueError:
            pass
        else:
            if bad.strip().lower() not in ("scaffold", "molecule", "random"):
                raise AssertionError(f"mode {bad!r} was accepted")
    print("  [15] an unknown mode raises rather than defaulting          OK")

    print("\nStage 10 data-split self-test passed.")


def _parse_args(argv: list) -> dict:
    """
    --random / --molecule / --mode NAME, --val-frac F, --test-frac F,
    --seed N, --rebuild.

    The mode flags are mutually exclusive; passing two is a typo, not a
    preference, and guessing which was meant would silently produce the wrong
    split.
    """
    named = [m for m, flag in (("random", "--random"),
                               ("molecule", "--molecule"),
                               ("scaffold", "--scaffold")) if flag in argv]
    if "--mode" in argv:
        i = argv.index("--mode")
        if i + 1 >= len(argv):
            raise SystemExit("--mode needs a value "
                             "(scaffold | molecule | random)")
        named.append(argv[i + 1].lower())
    if len(set(named)) > 1:
        raise SystemExit(f"conflicting split modes requested: "
                         f"{', '.join(sorted(set(named)))}")

    out: Dict[str, object] = {"rebuild": "--rebuild" in argv}
    if named:
        out["mode"] = named[0]
    for flag, key, cast in (("--val-frac", "val_frac", float),
                            ("--test-frac", "test_frac", float),
                            ("--seed", "seed", int)):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} needs a value")
            out[key] = cast(argv[i + 1])
    return out


def main(mode: str = None, val_frac: float = None, test_frac: float = None,
         seed: int = None, rebuild: bool = False) -> None:
    """Build (or refresh) the shared manifest from the real training pool."""
    from stage9_masked_property_finetune import collect_all_training_pairs

    mode      = (SPLIT_MODE if mode is None else mode).lower()
    val_frac  = VAL_FRAC   if val_frac  is None else val_frac
    test_frac = TEST_FRAC  if test_frac is None else test_frac
    seed      = SPLIT_SEED if seed      is None else seed

    blurb = {
        "scaffold": "whole Bemis-Murcko groups move together",
        "molecule": "one group per canonical SMILES, largest-first",
        "random":   "one group per canonical SMILES, SHUFFLED",
    }.get(mode, "?")

    print("\n" + "=" * 62)
    print(f"STAGE 10 -- SHARED TRAIN / VALIDATION SPLIT  [{mode}]")
    print("=" * 62)
    print(f"""
  Built once over every parent molecule in the training pool, cached, and
  read by Stage 10, 10.1, 10.2 and 10.4 so all four agree on what
  "validation" means.

    mode      : {mode}   ({blurb})
    val_frac  : {val_frac}
    test_frac : {test_frac}
    seed      : {seed}
    manifest  : {manifest_path(mode=mode)}
""")
    if mode != SPLIT_MODE:
        # Building a manifest the trainers will not read is a silent no-op,
        # and the only symptom would be validation curves that do not match
        # the mode just built. Say it here, once, with the fix.
        print(f"  NOTE: config.STAGE10_SPLIT_MODE is {SPLIT_MODE!r}, so the "
              f"trainers will read the\n        {SPLIT_MODE} manifest, not this "
              f"one. To train against this split, set\n"
              f"            STAGE10_SPLIT_MODE = {mode!r}\n"
              f"        in your config. Both manifests can coexist -- the mode "
              f"is in the filename.\n")

    pairs = collect_all_training_pairs(
        max_pairs=None,
        max_per_parent=None,      # the SPLIT is built over every parent, so it
    )                             # does not depend on the training cap
    if not pairs:
        print("  No pairs found. Run stage1a/stage1b first.")
        sys.exit(1)

    man = get_split([p for _, p in pairs], rebuild=rebuild, val_frac=val_frac,
                    test_frac=test_frac, seed=seed, mode=mode)
    tr, va, te = split_pairs(pairs, man=man)
    print()
    for line in describe(man, (tr, va, te)):
        print(line)
    print()


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main(**_parse_args(sys.argv))


# ════════════════════════════════════════════════════════════════════════════
#  THE VALIDATION PASS
# ════════════════════════════════════════════════════════════════════════════

VAL_MAX_PAIRS   = getattr(config, "STAGE10_VAL_MAX_PAIRS", 2000)
SELECT_BEST_VAL = getattr(config, "STAGE10_SELECT_BEST_VAL", True)

# The per-epoch series a validation pass contributes, on top of whatever the
# stage already records. Kept here rather than in each trainer so that the four
# implementations cannot drift into recording different things and calling them
# by the same name.
VAL_CORE_KEYS = ("val_n", "val_loss_mean", "val_best_loss_mean",
                 "val_cand_valid_rate", "val_best_valid_rate",
                 "val_fallback_rate")


def val_history_keys(loss_terms: Sequence[str]) -> Tuple[str, ...]:
    """VAL_CORE_KEYS plus one val_ series per loss term of the calling stage."""
    return VAL_CORE_KEYS + tuple("val_" + t for t in loss_terms)


def validation_pass(val_pairs, loss_fn, model, loss_terms: Sequence[str],
                    batch_size: int = 16, max_pairs: int = None,
                    seed: int = 0, log=None) -> Dict[str, float]:
    """
    Score the held-out fold under the stage's own best-of-K procedure, without
    gradients, and return its metrics as val_-prefixed history entries.

    `loss_fn(batch) -> (loss, stats)` is the calling stage's batch loss with
    everything else already bound. Taking a closure rather than the function
    itself is what lets one implementation serve Stage 10 (no pool), 10.1
    (pool) and 10.2/10.4 (pool, AMP, batched forward) without this module
    knowing anything about their signatures.

    THE RNG STREAM IS SAVED AND RESTORED AROUND THE PASS, and that is not
    tidiness. Stage 10 and Stage 10.1 claim bit-identical execution, and every
    trainer here checkpoints the RNG so a resume continues the same stream.
    A validation pass draws candidates, so it consumes the torch RNG; left
    alone it would shift every subsequent training draw and a run WITH
    validation would diverge from the same run without it. Wrapping the pass
    makes it observationally inert on training.

    The model is put in eval() for the pass -- dropout off, so the number is a
    property of the weights rather than of a particular dropout mask -- and
    returned to train() afterwards. Sampling stays stochastic, which is
    intended: we are measuring the quality of what the model GENERATES, not
    reconstructing a fixed target.
    """
    import torch

    log = log or tqdm.write
    max_pairs = VAL_MAX_PAIRS if max_pairs is None else max_pairs

    pairs = list(val_pairs)
    if max_pairs and len(pairs) > max_pairs:
        # Deterministic subsample: the SAME validation molecules every epoch
        # and every run, or the curve moves for reasons that are not the model.
        pairs = random.Random(seed).sample(pairs, max_pairs)
    if not pairs:
        return {}

    rng_state = (random.getstate(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if torch.cuda.is_available()
                 else None)
    was_training = model.training
    model.eval()

    agg = {k: 0.0 for k in ("loss", "best_loss", "fallback", "cand_valid",
                            "cand_total", "n", "best_valid")}
    for t in loss_terms:
        agg[t] = 0.0
    n_steps = 0

    try:
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                loss, stats = loss_fn(pairs[i:i + batch_size])
                if stats.get("n", 0) == 0:
                    continue
                agg["loss"] += float(loss)
                for k in ("best_loss", "fallback", "cand_valid", "cand_total",
                          "n", "best_valid"):
                    agg[k] += stats.get(k, 0.0)
                for t in loss_terms:
                    agg[t] += stats.get(t, 0.0)
                n_steps += 1
    finally:
        if was_training:
            model.train()
        random.setstate(rng_state[0])
        torch.set_rng_state(rng_state[1])
        if rng_state[2] is not None:
            torch.cuda.set_rng_state_all(rng_state[2])

    n_mol = max(agg["n"], 1.0)
    out = {
        "val_n": agg["n"],
        "val_loss_mean": agg["loss"] / max(n_steps, 1),
        "val_best_loss_mean": agg["best_loss"] / n_mol,
        "val_cand_valid_rate": agg["cand_valid"] / max(agg["cand_total"], 1.0),
        "val_best_valid_rate": agg["best_valid"] / n_mol,
        "val_fallback_rate": agg["fallback"] / n_mol,
    }
    for t in loss_terms:
        out["val_" + t] = agg[t] / n_mol
    return out


def format_val_line(row: Dict[str, float]) -> str:
    """The one-line epoch summary of a validation pass."""
    if not row or not row.get("val_n"):
        return ""
    return (f"    held-out   -- loss={row['val_best_loss_mean']:.3f}  "
            f"candidate_validity={row['val_cand_valid_rate']:.1%}  "
            f"target_validity={row['val_best_valid_rate']:.1%}  "
            f"fallback={row['val_fallback_rate']:.1%}  "
            f"(n={int(row['val_n'])})")


def is_best_val(history: Dict[str, list],
                key: str = "val_best_loss_mean") -> bool:
    """
    True when the epoch just appended is the best validation epoch so far.

    Lower is better -- `key` is a composite LOSS. A NaN (no validation fold, or
    every batch skipped) is never best, so a run without validation data keeps
    whatever selection behaviour it had.
    """
    series = [v for v in (history.get(key) or []) if v == v]
    if not series:
        return False
    return series[-1] <= min(series)


def split_pairs_by_source(pairs_by_source: Dict[str, List[Tuple[str, str]]],
                          man: Dict[str, object] = None, path: str = None,
                          log=None) -> Dict[str, List[Tuple[str, str]]]:
    """
    Turn {"stage1a": [...], "stage1b": [...]} into
    {"stage1a_train": [...], "stage1a_heldout": [...], ...} for the final
    property report.

    Splitting the SOURCE KEYS rather than adding a parameter to the plotting
    code is deliberate: plot_property_report already draws one series per key,
    so train and held-out appear side by side on every panel with no change to
    the figure code, and the train/held-out gap -- which is the memorisation
    measurement -- is read straight off the plot.

    Empty sides are dropped, so a source with no held-out molecules (possible
    when a source is tiny) contributes one key rather than an empty panel.
    """
    log = log or tqdm.write
    if man is None:
        every = [p for pairs in pairs_by_source.values() for _, p in pairs]
        man = get_split(every, path=path, log=log)

    out: Dict[str, List[Tuple[str, str]]] = {}
    for source, pairs in pairs_by_source.items():
        held, train = [], []
        for pair in pairs:
            (train if fold_of(pair[1], man) == "train" else held).append(pair)
        if train:
            out[f"{source}_train"] = train
        if held:
            out[f"{source}_heldout"] = held
    return out

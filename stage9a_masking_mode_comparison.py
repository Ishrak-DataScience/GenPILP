# -*- coding: utf-8 -*-
"""
stage9a_masking_mode_comparison.py
==================================
Stage 9a's measurement, applied to the question Stage 1b exists to ask:
does it MATTER which part of the ligand you mask?

Three arms, one figure
----------------------
Every arm covers the same molecules, blanks out the same number of <mask>
tokens, and is refilled by the same vanilla ChemBERTa in the same way. The only
thing that differs is WHICH atoms were eligible to be masked:

  1. INTERACTING      the atoms PLIP reports as making protein-ligand contacts
                      -- the pharmacophore. The model must re-invent the part
                      that does the binding.
  2. NON-INTERACTING  every other heavy atom -- the scaffold around the
                      contacts. The model must re-invent the frame while the
                      binding atoms are handed to it.
  3. RANDOM           atoms drawn uniformly from the whole molecule, ignoring
                      PLIP entirely. The control arm: it is what "masking"
                      alone buys, with no interaction information in it. Any
                      gap between arms 1/2 and this one is what the PLIP
                      annotation is worth.

and four criteria per arm, the four this comparison was asked for:

  * RDKit validity      -- % of refilled molecules RDKit can parse at all,
                           over EVERY attempt. The only panel whose
                           denominator is all generations.
  * Drug-likeness       -- QED, 0-1, over the valid ones.
  * Synthetic accessib. -- SA score on its native 1-10 scale (1 = easy to
                           make), over the valid ones.
  * Toxicity            -- PAINS/Brenk structural-alert hit rate, plus the
                           Tox21 classifier panel when
                           config.STAGE9_TOX21_MODEL_DIR is set.

Scaffold novelty is measured and written to the CSV but left OFF the figure:
four criteria were asked for, and novelty against the parent means something
different per arm (masking the contacts changes more of the fingerprint than
masking the scaffold does, before any model is involved). Drop "novelty" from
OMIT_PANELS to draw it.

Where the three arms come from
-------------------------------
ONE Stage 1b summary, not three runs. stage1_mask_calculation.py builds the
two PLIP arms as exact complements of each other:

    attractive_indices = <atoms PLIP reports as interacting>
    masked_indices     = (set(range(n_atoms)) - attractive_indices)   # mode 2
                       or attractive_indices                          # mode 1

so whichever mode produced the summary on disk, the OTHER arm is recoverable
from it exactly -- not approximated. The summary's own `masking_mode` column
says which set was stored, and this script reads it per row rather than
assuming; a row whose mode it does not recognise is dropped and counted. The
shipped summary is "non-attractive", so the interacting arm is the complement
there. If you would rather generate both arms directly, run Stage 1b twice
(mode 1 and mode 2) -- the numbers must come out the same, because this is the
same set arithmetic that script does.

The matched mask budget (the part that makes this a comparison)
----------------------------------------------------------------
The arms must be equally HARD, or the panels compare difficulty rather than
chemistry. What sets the difficulty is the number of <mask> tokens the model
is asked to fill -- that is literally all it sees -- so that is what is
matched, per molecule:

    K = min(floor(STAGE9_MASK_PERCENT% x n_bpe_tokens),
            <max tokens the interacting pool can cover>,
            <max tokens the non-interacting pool can cover>)

and every arm masks EXACTLY K tokens, drawing atoms from its own pool until it
gets there. The molecule is dropped from ALL THREE arms if any of them cannot
land on K exactly, so the arms never diverge onto different molecule sets.

Matching tokens rather than ATOMS is a deliberate choice, and it was the wrong
way round first. Masking a fixed number of atoms from each region gave, on a
60-molecule trial, 7.1 atoms -> 7.5 masks in the interacting arm but 13.6 in
the non-interacting one: contacts are mostly single-atom BPE tokens, scaffold
atoms are not. That is a 1.8x difference in how much of the string was blanked
out, which is more than enough to produce the entire validity gap on its own.
Under token matching the model is handed the same number of blanks in every
arm and only their POSITION differs -- so a gap between arms is attributable
to the region.

The consequence, stated on the figure rather than hidden: the arms mask
DIFFERENT NUMBERS OF ATOMS (fewer in the non-interacting arm, since its atoms
cover more tokens each). `--match atoms` runs the other design if you want the
comparison the other way round.

Molecules with no interacting atoms at all (PLIP resolved no contact for that
ligand instance) cannot form arm 1 and are dropped from the comparison rather
than counted as "nothing masked".

What gets written (into config.STAGE9A_DIR)
--------------------------------------------
  stage9a_masking_mode_comparison.png   the four-panel, three-arm figure
  stage9a_masking_mode_per_molecule.csv one row per attempt, with the arm, the
                                        matched budget, the mask-token count
                                        and every measured property, so the
                                        figure can be redrawn or the numbers
                                        recomputed without another GPU pass

`--match atoms` writes the same two under *_atom_matched names rather than
overwriting these: the two designs answer different questions, and a figure
sitting under a caption that describes the other one is worse than no figure.

Usage
-----
  python stage9a_masking_mode_comparison.py
  python stage9a_masking_mode_comparison.py --limit 2000
  python stage9a_masking_mode_comparison.py --limit none --seed 7
  python stage9a_masking_mode_comparison.py --match atoms     (the other design)
  python stage9a_masking_mode_comparison.py --parent-reference off
  python stage9a_masking_mode_comparison.py --test
"""

from __future__ import annotations

# See the Stage 10 scripts: torchao logs a register_constant() deprecation as
# it is imported, and a filter installed afterwards has nothing to catch.
# Guarded so a checkout missing the module still runs.
try:
    import quiet_torch_logs  # noqa: F401
except ImportError:
    pass

import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import config
from stage1_mask_calculation import (
    _clean_smiles_atom_spans,
    _smiles_output_atom_order,
    mask_atoms_in_smiles_token_level,
)
from stage9_masked_property_finetune import (
    MAX_MODEL_TOKENS,
    PROPERTY_CSV_FIELDS,
    TEMPERATURE,
    TOP_K,
    _TOX21_AVAILABLE,
    _TOX21_MODEL_DIR,
    _canonical_parent,
    _read_csv_rows_matching,
    _remask_seed,
    evaluate_parent_property_records,
    evaluate_property_records,
    format_property_summary,
    get_chemberta_tokenizer,
    plot_property_report,
)
from stage9a_masked_property_without_finetuning import load_chemberta_base

try:
    from tqdm import tqdm
except ImportError:                                   # pragma: no cover
    from tqdm_compat import tqdm  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════════
#  THE THREE ARMS
# ════════════════════════════════════════════════════════════════════════════

# What is held equal across the arms. Tokens is the default and the reason is
# in the module docstring: the model sees masks, not atoms, so matching atoms
# leaves the arms at different difficulties and the panels stop being about
# the region.
MATCH_TOKENS = "tokens"
MATCH_ATOMS  = "atoms"

INTERACTING     = "mask_interacting"
NON_INTERACTING = "mask_non_interacting"
RANDOM_ATOMS    = "mask_random"
PARENT          = "parent_reference"

ARM_ORDER = (INTERACTING, NON_INTERACTING, RANDOM_ATOMS)
PLOT_ORDER = ARM_ORDER + (PARENT,)

ARM_LABELS = {
    INTERACTING:     "PLIP interacting atoms masked",
    NON_INTERACTING: "Non-interacting atoms masked",
    RANDOM_ATOMS:    "Random atoms masked (control)",
    PARENT:          "Parent molecule, pre-mask (reference)",
}
SHORT_LABELS = {
    INTERACTING:     "Interacting",
    NON_INTERACTING: "Non-interacting",
    RANDOM_ATOMS:    "Random",
    PARENT:          "Parent",
}
# Categorical slots 1-3 of the same validated, colorblind-safe palette Stage 9
# draws its two source colours from (blue/orange are slots 1/2 there). Those
# first three slots are the set that clears the CVD separation floor on EVERY
# pair, not just adjacent ones -- which is what a three-series panel needs,
# since all three bars sit side by side in the validity and alert panels.
#
# The parent reference wears MUTED INK rather than a fourth slot on purpose:
# it is not a fourth condition, it is the line the three conditions are read
# against, and a categorical hue would present it as a competitor.
ARM_COLORS = {
    INTERACTING:     "#2a78d6",
    NON_INTERACTING: "#eb6834",
    RANDOM_ATOMS:    "#1baf7a",
    PARENT:          "#898781",
}
# Parent and the arms are the SAME molecules, so the parent series is told
# apart by texture rather than by being given its own colour identity.
PARENT_HATCH = "///"

# Novelty is measured but not drawn -- see the module docstring.
OMIT_PANELS = ("novelty",)

FIG_NAME = {
    MATCH_TOKENS: "stage9a_masking_mode_comparison.png",
    MATCH_ATOMS:  "stage9a_masking_mode_comparison_atom_matched.png",
}
CSV_NAME = {
    MATCH_TOKENS: "stage9a_masking_mode_per_molecule.csv",
    MATCH_ATOMS:  "stage9a_masking_mode_per_molecule_atom_matched.csv",
}

# Extra per-attempt columns this comparison needs and Stage 9's own CSV has no
# place for: the arm, what the matched budget came out at, and how many mask
# TOKENS that turned into (the confound the footer reports).
CSV_FIELDS = list(PROPERTY_CSV_FIELDS) + [
    "n_mask_atoms", "n_mask_tokens", "n_heavy_atoms",
    "n_interacting", "n_non_interacting", "pdb_id", "ligand",
]

# The summary's masking_mode column -> which set its masked_atom_indices holds.
_STORED_IS_INTERACTING = {"attractive": True, "non-attractive": False}


# ════════════════════════════════════════════════════════════════════════════
#  BUILDING THE THREE ARMS FROM ONE STAGE 1b SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def atom_token_map(smiles: str, tokenizer) -> Optional[Dict[int, frozenset]]:
    """
    {atom index -> the BPE tokens that atom would mask}, for the canonical
    form of `smiles`.

    This is the same atom -> covering-token relation
    mask_atoms_in_smiles_token_level applies, lifted out so a candidate atom
    set can be PRICED IN TOKENS without building the masked string. Selecting
    atoms to hit an exact token count needs one such price per candidate atom;
    going through the masker for each would re-parse and re-canonicalise the
    molecule every time, which is ~1 ms a go against set arithmetic's ~1 us.

    Returns None when the mapping cannot be built (RDKit refuses the SMILES,
    the tokenizer offers no offsets, span/atom counts disagree) -- the caller
    drops the molecule, exactly as it would if the masker had raised.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    clean = Chem.MolToSmiles(mol, canonical=True)
    try:
        order = _smiles_output_atom_order(mol)
        spans = _clean_smiles_atom_spans(clean)
    except Exception:
        return None
    if len(spans) != len(order):
        return None

    enc     = tokenizer(clean, return_offsets_mapping=True, add_special_tokens=False)
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"] if int(s) < int(e)]
    if not offsets:
        return None

    mapping: Dict[int, frozenset] = {}
    for atom_idx, (b, e) in zip(order, spans):
        mapping[atom_idx] = frozenset(
            t for t, (ts, te) in enumerate(offsets) if ts < e and te > b)
    return mapping


def _select_atoms_for_tokens(
    pool:      Sequence[int],
    token_of:  Dict[int, frozenset],
    k_tokens:  int,
    seed_key:  str,
    base_seed: int,
) -> Optional[List[int]]:
    """
    Pick atoms from `pool` until they mask EXACTLY `k_tokens` tokens, or give
    up and return None.

    Atoms are considered in one deterministic shuffled order, and an atom that
    would push the count PAST k_tokens is skipped rather than ending the walk:
    a single bracket atom can straddle two tokens, so stopping at the first
    overshoot would leave the arm short of a target another arm hit exactly,
    and the two would no longer be comparable. Skipping lets a later,
    cheaper atom close the gap.

    Masked tokens are a monotone function of the atom set (a token is masked
    if ANY of its atoms is), which is what makes the walk safe: the count only
    ever rises, so a running set is enough and nothing needs re-checking.
    """
    if k_tokens <= 0:
        return None
    rng   = random.Random(_remask_seed(base_seed, seed_key))
    order = sorted(pool)
    rng.shuffle(order)

    chosen: List[int] = []
    masked: set = set()
    for atom in order:
        tokens = token_of.get(atom)
        if not tokens or tokens <= masked:
            continue                       # free atoms add nothing to price
        if len(masked | tokens) > k_tokens:
            continue                       # overshoots -- try a smaller atom
        masked |= tokens
        chosen.append(atom)
        if len(masked) == k_tokens:
            return sorted(chosen)
    return None


def _max_tokens(pool: Sequence[int], token_of: Dict[int, frozenset]) -> int:
    """Tokens masked if EVERY atom in `pool` were masked -- the arm's ceiling."""
    covered: set = set()
    for atom in pool:
        covered |= token_of.get(atom, frozenset())
    return len(covered)


def _mask_n_atoms(
    smiles:    str,
    pool:      Sequence[int],
    n_mask:    int,
    tokenizer,
    seed_key:  str,
    base_seed: int,
) -> Tuple[str, int]:
    """
    Mask exactly `n_mask` atoms drawn from `pool`, returning
    (masked_smiles, n_mask_tokens) or ("", 0) if the result is unusable.

    The explicit count is the whole difference from Stage 9's
    _remask_from_pool, which derives its own count from the mask percentage
    and the pool size. Here the count is decided ONCE for the molecule and
    handed to every arm -- deriving it per arm is exactly the confound the
    matched budget exists to remove.

    The pool is sorted before sampling so the draw depends on the SET of
    eligible atoms and the seed, never on the order they arrived in.
    """
    if n_mask <= 0 or n_mask > len(pool):
        return "", 0
    rng     = random.Random(_remask_seed(base_seed, seed_key))
    sampled = sorted(rng.sample(sorted(pool), n_mask))
    try:
        masked = mask_atoms_in_smiles_token_level(smiles, sampled, tokenizer)
    except Exception:
        return "", 0
    if not masked or tokenizer.mask_token not in masked:
        return "", 0
    # Masking can LENGTHEN a molecule in tokens (each <mask> is its own token,
    # and masking splits BPE tokens that used to merge), so the masked string
    # gets its own window check rather than inheriting the parent's.
    if len(tokenizer(masked, add_special_tokens=True)["input_ids"]) > MAX_MODEL_TOKENS:
        return "", 0
    return masked, masked.count(tokenizer.mask_token)


def _build_token_matched(
    smiles:    str,
    pools:     Dict[str, List[int]],
    target:    int,
    tokenizer,
    seed_key:  str,
    base_seed: int,
    stats:     dict,
) -> Optional[Dict[str, Tuple[str, int, int]]]:
    """
    Build all three arms at ONE shared <mask>-token count, or return None and
    count why not.

    The count starts at `target` (STAGE9_MASK_PERCENT% of the molecule's BPE
    tokens) capped by what the smallest pool can actually cover, then falls to
    whatever every arm managed if some arm could not land on it exactly.
    Lowering the shared target is what makes an exact match reachable: a
    bracket atom that straddles two tokens can make one value unhittable while
    the value below it is fine, and an arm left one token short of the others
    is no longer the same task.

    Returns {arm: (masked_smiles, n_mask_tokens, n_mask_atoms)}. The atom
    counts DIFFER between arms -- that is the price of matching tokens, and
    the figure footer reports it.
    """
    token_of = atom_token_map(smiles, tokenizer)
    if token_of is None:
        stats["token_map_failed"] = stats.get("token_map_failed", 0) + 1
        return None

    ceiling = min(_max_tokens(pool, token_of) for pool in pools.values())
    k = min(target, ceiling)
    if k < 1:
        stats["budget_zero"] += 1
        return None

    # At most a few rounds: each one lowers k to a value that at least one arm
    # demonstrably reached, so it converges rather than searching.
    for _ in range(4):
        picked = {arm: _select_atoms_for_tokens(pool, token_of, k, seed_key, base_seed)
                  for arm, pool in pools.items()}
        achieved = {arm: (len(set().union(*(token_of[a] for a in atoms)))
                          if atoms else 0)
                    for arm, atoms in picked.items()}
        if all(v == k for v in achieved.values()) and None not in picked.values():
            break
        k = min(v for v in achieved.values())
        if k < 1:
            stats["token_match_failed"] = stats.get("token_match_failed", 0) + 1
            return None
    else:
        stats["token_match_failed"] = stats.get("token_match_failed", 0) + 1
        return None

    built: Dict[str, Tuple[str, int, int]] = {}
    for arm, atoms in picked.items():
        if atoms is None:                    # cannot happen past the loop above
            return None                      # -- kept so the types say so too
        try:
            masked = mask_atoms_in_smiles_token_level(smiles, atoms, tokenizer)
        except Exception:
            stats["mask_failed"] += 1
            return None
        if not masked or tokenizer.mask_token not in masked:
            stats["mask_failed"] += 1
            return None
        if len(tokenizer(masked, add_special_tokens=True)["input_ids"]) > MAX_MODEL_TOKENS:
            stats["over_length_masked"] = stats.get("over_length_masked", 0) + 1
            return None
        n_mask_tokens = masked.count(tokenizer.mask_token)
        if n_mask_tokens != k:
            # The priced count and the built string disagree -- the map and
            # the masker have drifted apart. Drop the molecule rather than
            # report a match that isn't one.
            stats["token_count_mismatch"] = stats.get("token_count_mismatch", 0) + 1
            return None
        built[arm] = (masked, n_mask_tokens, len(atoms))
    return built


def build_masking_mode_pairs(
    stage1b_dir:     str  = None,
    max_molecules:   int  = None,
    sample_seed:     int  = None,
    mask_percent:    float = None,
    mask_seed:       int  = None,
    dedup_by_parent: bool = None,
    match:           str  = MATCH_TOKENS,
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, List[dict]], dict]:
    """
    Build all three arms from the Stage 1b summary, as PAIRED TRIPLES.

    Returns (pairs_by_arm, meta_by_arm, stats), where meta_by_arm[arm] runs
    PARALLEL to pairs_by_arm[arm] -- one bookkeeping dict per pair, in the same
    order (budget, token count, pool sizes, which PDB entry it came from).

    Parallel lists rather than a {(arm, masked_smiles): meta} dict on purpose:
    two different parents can mask down to the SAME string (small ligands where
    the budget covers nearly everything), and keying on it silently merged
    them -- which showed up as the arms reporting mean token counts that
    differed in the second decimal when they are equal by construction.
    evaluate_property_records returns one record per pair in pair order, so the
    index is an exact join where the string was a lossy one.

    `match` is what is held equal across the arms -- MATCH_TOKENS (the
    default: the same number of <mask> tokens per molecule, so the arms are
    equally hard and only the masked REGION differs) or MATCH_ATOMS (the same
    number of atoms, so the same amount of chemistry is removed). See the
    module docstring for why tokens is the default.

    A molecule enters the comparison only if ALL THREE arms could be built for
    it. Dropping the triple rather than the failing arm is what keeps the
    panels comparable: three arms over three different molecule sets would be
    three measurements, not one comparison. Every drop is counted and ends up
    in the figure footer.
    """
    if match not in (MATCH_TOKENS, MATCH_ATOMS):
        raise ValueError(f"match must be {MATCH_TOKENS!r} or {MATCH_ATOMS!r}, "
                         f"got {match!r}")
    stage1b_dir = (stage1b_dir
                   or getattr(config, "STAGE9_9A_STAGE1B_DATA_DIR", "")
                   or config.STAGE1B_PLIP_MASK_DIR)
    if mask_percent is None:
        mask_percent = config.STAGE9_MASK_PERCENT
    if mask_seed is None:
        mask_seed = config.STAGE9_MASK_SEED
    if sample_seed is None:
        sample_seed = getattr(config, "STAGE9_PAIR_SAMPLE_SEED", 42)
    if dedup_by_parent is None:
        dedup_by_parent = getattr(config, "STAGE9_EVAL_DEDUP_BY_PARENT", True)

    rows = _read_csv_rows_matching(
        stage1b_dir, "stage1b_large_scale_plip_mask_summary.csv")
    stats: dict = {
        "location": stage1b_dir, "rows": len(rows), "status_not_ok": 0,
        "unknown_mode": 0, "missing_field": 0, "bad_pool": 0,
        "invalid_parent": 0, "pool_out_of_range": 0, "over_length_parent": 0,
        "duplicate_parents": 0, "no_interacting_atoms": 0,
        "no_non_interacting_atoms": 0, "budget_zero": 0, "mask_failed": 0,
        "molecules": 0, "modes_seen": {},
    }
    if not rows:
        return {}, {}, stats

    tokenizer = get_chemberta_tokenizer()

    # ── 1. usable rows, deduped to one instance per distinct parent ────────
    # Stage 1b writes one row per ligand INSTANCE, so a ligand solved in 400
    # PDB entries would otherwise carry 400x the weight of a ligand solved
    # once -- and PDB instance counts follow crystallography, not chemistry.
    # Same rule and same reasoning as Stage 9's _dedup_pairs_by_parent; it is
    # applied to ROWS here because the three arms must stay paired, which they
    # cannot be if each arm dedups its own pair list independently.
    usable: List[dict] = []
    seen: set = set()
    for row in rows:
        if row.get("status") != "ok":
            stats["status_not_ok"] += 1
            continue
        mode = (row.get("masking_mode") or "").strip()
        stats["modes_seen"][mode] = stats["modes_seen"].get(mode, 0) + 1
        if mode not in _STORED_IS_INTERACTING:
            stats["unknown_mode"] += 1
            continue
        smiles = (row.get("smiles") or "").strip()
        if not smiles:
            stats["missing_field"] += 1
            continue
        if dedup_by_parent:
            key = _canonical_parent(smiles)
            if key in seen:
                stats["duplicate_parents"] += 1
                continue
            seen.add(key)
        usable.append(row)

    # ── 2. sample BEFORE masking, so a smaller limit is a shorter run ──────
    # Drawn from the shared row list, which is what makes --limit mean "this
    # many molecules in every arm" rather than "this many per arm, possibly
    # different ones".
    stats["candidate_molecules"] = len(usable)
    if max_molecules and len(usable) > max_molecules:
        rng    = random.Random(_remask_seed(sample_seed, "masking_mode_comparison"))
        usable = rng.sample(usable, max_molecules)
        stats["sampled_from"] = stats["candidate_molecules"]
        stats["sample_seed"]  = sample_seed

    # ── 3. one triple per molecule, or nothing ────────────────────────────
    pairs_by_arm: Dict[str, List[Tuple[str, str]]] = {a: [] for a in ARM_ORDER}
    meta_by_arm:  Dict[str, List[dict]] = {a: [] for a in ARM_ORDER}

    for row in tqdm(usable, desc="  Building the three arms", unit="mol"):
        smiles = (row.get("smiles") or "").strip()
        mol    = Chem.MolFromSmiles(smiles)
        if mol is None:
            stats["invalid_parent"] += 1
            continue
        n_atoms = mol.GetNumAtoms()
        try:
            stored = {int(i) for i in json.loads(row.get("masked_atom_indices") or "[]")}
        except (json.JSONDecodeError, TypeError, ValueError):
            stats["bad_pool"] += 1
            continue
        if any(i < 0 or i >= n_atoms for i in stored):
            # The stored indices address a different molecule than the stored
            # SMILES does -- the complement would be meaningless, so the row
            # goes rather than quietly producing a wrong "interacting" set.
            stats["pool_out_of_range"] += 1
            continue

        everything = set(range(n_atoms))
        if _STORED_IS_INTERACTING[row["masking_mode"].strip()]:
            interacting = stored
        else:
            interacting = everything - stored
        non_interacting = everything - interacting

        if not interacting:
            stats["no_interacting_atoms"] += 1
            continue
        if not non_interacting:
            stats["no_non_interacting_atoms"] += 1
            continue

        n_tokens = len(tokenizer(smiles, add_special_tokens=False)["input_ids"])
        if n_tokens + 2 > MAX_MODEL_TOKENS:           # +2 for <s> / </s>
            stats["over_length_parent"] += 1
            continue

        seed_key = (f"{row.get('pdb_id','')}:{row.get('resname','')}:"
                    f"{row.get('chain','')}:{row.get('resseq','')}")
        pools = {
            INTERACTING:     sorted(interacting),
            NON_INTERACTING: sorted(non_interacting),
            RANDOM_ATOMS:    sorted(everything),
        }
        target = int(mask_percent / 100.0 * n_tokens)        # floor

        if match == "atoms":
            budget = min(target, len(interacting), len(non_interacting))
            if budget < 1:
                stats["budget_zero"] += 1
                continue
            built: Optional[Dict[str, Tuple[str, int, int]]] = {}
            for arm, pool in pools.items():
                masked, n_mask_tokens = _mask_n_atoms(
                    smiles, pool, budget, tokenizer, seed_key, mask_seed)
                if not masked:
                    break
                built[arm] = (masked, n_mask_tokens, budget)
            if len(built) != len(pools):
                stats["mask_failed"] += 1
                continue
        else:
            built = _build_token_matched(
                smiles, pools, target, tokenizer, seed_key, mask_seed, stats)
            if built is None:
                continue

        for arm, (masked, n_mask_tokens, n_mask_atoms) in built.items():
            pairs_by_arm[arm].append((masked, smiles))
            meta_by_arm[arm].append({
                "n_mask_atoms":      n_mask_atoms,
                "n_mask_tokens":     n_mask_tokens,
                "n_heavy_atoms":     n_atoms,
                "n_interacting":     len(interacting),
                "n_non_interacting": len(non_interacting),
                "pdb_id":            row.get("pdb_id", ""),
                "ligand":            seed_key,
            })
        stats["molecules"] += 1

    pairs_by_arm = {a: p for a, p in pairs_by_arm.items() if p}
    meta_by_arm = {a: m for a, m in meta_by_arm.items() if m}
    return pairs_by_arm, meta_by_arm, stats


# ════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ════════════════════════════════════════════════════════════════════════════

def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def format_comparison_footer(
    stats:        dict,
    meta_by_arm:  Dict[str, List[dict]],
    mask_percent: float,
    match:        str = MATCH_TOKENS,
) -> str:
    """
    The footer states the two things a reader needs to trust the panels: that
    the arms cover the SAME molecules at the SAME matched budget, and what the
    quantity that could NOT be matched came out at per arm -- mask tokens when
    matching atoms, masked atoms when matching tokens. Whichever it is, it is
    the confound, and it goes on the figure rather than in a note somewhere.
    """
    per_arm = []
    for arm in ARM_ORDER:
        metas  = meta_by_arm.get(arm) or []
        if not metas:
            continue
        atoms  = _mean([m["n_mask_atoms"] for m in metas])
        tokens = _mean([m["n_mask_tokens"] for m in metas])
        per_arm.append(f"{SHORT_LABELS[arm]} {atoms:.1f} atoms -> {tokens:.1f} masks")

    dropped = {k: v for k, v in stats.items()
               if k in ("status_not_ok", "unknown_mode", "missing_field",
                        "bad_pool", "invalid_parent", "pool_out_of_range",
                        "over_length_parent", "duplicate_parents",
                        "no_interacting_atoms", "no_non_interacting_atoms",
                        "budget_zero", "mask_failed", "token_map_failed",
                        "token_match_failed", "token_count_mismatch",
                        "over_length_masked") and v}
    drop_txt = ", ".join(f"{k}: {v}" for k, v in dropped.items()) or "none"

    if match == MATCH_TOKENS:
        headline = (
            f"{stats.get('molecules', 0)} molecule(s) per arm, the SAME molecules "
            f"in all three: one Stage 1b row per distinct parent, every arm "
            f"masking the SAME NUMBER OF <mask> TOKENS "
            f"(min({mask_percent:g}% of the molecule's BPE tokens, what each "
            f"region can cover)) with the same seed -- so the arms are equally "
            f"hard and only the masked REGION differs.")
        caveat = ("  (tokens are matched by construction; the ATOM counts that "
                  "took each arm there are not, and are stated here for that "
                  "reason -- contacts are mostly one token per atom, scaffold "
                  "atoms are not).")
    else:
        headline = (
            f"{stats.get('molecules', 0)} molecule(s) per arm, the SAME molecules "
            f"in all three: one Stage 1b row per distinct parent, every arm "
            f"masking the SAME NUMBER OF ATOMS "
            f"(min({mask_percent:g}% of BPE tokens, |interacting|, "
            f"|non-interacting|)) with the same seed.")
        caveat = ("  (atoms are matched by construction; the mask-TOKEN counts "
                  "that follow from them are NOT, so the arms are not equally "
                  "hard -- read the validity panel against these numbers).")

    lines = [
        headline,
        f"Mean mask size per arm: " + "; ".join(per_arm) + caveat,
        f"Read from {stats.get('location', '?')} -- {stats.get('rows', 0)} row(s); "
        f"dropped ({drop_txt}).",
    ]
    if stats.get("sampled_from"):
        lines.append(
            f"Sampled {stats.get('molecules', 0)} of {stats['sampled_from']} "
            f"eligible molecules (seed {stats.get('sample_seed')}); set "
            f"--limit none to use them all.")
    return "\n".join(lines)


def write_comparison_csv(
    records_by_arm: Dict[str, List[Dict[str, object]]],
    meta_by_arm:    Dict[str, List[dict]],
    csv_path:       str,
) -> str:
    """
    One row per attempt, with the mask bookkeeping joined back on BY INDEX --
    evaluate_property_records emits one record per pair in pair order, and
    meta_by_arm was built in that same order. Joining on the masked SMILES
    instead would merge the occasional pair of small ligands that mask down to
    the same string.

    Stage 9's own writer is not reused because it writes a fixed column set
    with no room for the budget or the token count -- and without those the
    figure's central claim (that the arms were matched) cannot be re-checked
    from the CSV.

    Parent rows carry blank mask columns: nothing was masked to produce them,
    and copying an arm's budget onto them would read as though something was.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for arm in PLOT_ORDER:
            records = records_by_arm.get(arm, [])
            metas   = meta_by_arm.get(arm) or []
            if metas and len(metas) != len(records):
                # Should be impossible; say so rather than silently writing a
                # CSV whose budget columns belong to other molecules.
                tqdm.write(f"  WARNING: {arm} has {len(records)} record(s) but "
                           f"{len(metas)} mask entr(ies) -- mask columns left "
                           f"blank for this arm.")
                metas = []
            for i, rec in enumerate(records):
                meta = metas[i] if i < len(metas) else {}
                row  = {**rec, **meta}
                writer.writerow({
                    k: ("" if row.get(k) is None else row.get(k))
                    for k in CSV_FIELDS
                })
    tqdm.write(f"  Per-molecule records : {csv_path}")
    return csv_path


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(
    max_molecules:    int  = None,
    sample_seed:      int  = None,
    parent_reference: bool = True,
    match:            str  = MATCH_TOKENS,
) -> None:
    if max_molecules is None:
        max_molecules = getattr(config, "STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE", None)
    mask_percent = config.STAGE9_MASK_PERCENT
    matched_note = ("the SAME number of <mask> TOKENS per molecule"
                    if match == MATCH_TOKENS else
                    "the SAME number of ATOMS per molecule")

    print("\n" + "=" * 68)
    print("STAGE 9a -- MASKING MODE COMPARISON (ChemBERTa, NO fine-tuning)")
    print("=" * 68)
    print(f"""
  Three arms over the SAME molecules, refilled by the same vanilla ChemBERTa
  ({config.CHEMBERTA_MODEL}), measured by the same code:

    1. {ARM_LABELS[INTERACTING]}
    2. {ARM_LABELS[NON_INTERACTING]}
    3. {ARM_LABELS[RANDOM_ATOMS]}

  Every arm masks {matched_note},
  targeting {mask_percent:g}% of the molecule's BPE tokens and capped by what
  the smaller region can supply, with the same per-molecule seed -- so the
  eligible REGION is the only difference between them. A molecule that cannot
  meet that budget in all three arms is dropped from all three.

  Panels: RDKit validity, QED (drug-likeness), synthetic accessibility, and
  toxicity ({"PAINS/Brenk alerts + Tox21" if _TOX21_AVAILABLE else "PAINS/Brenk structural alerts"}).

  Tox21 classifier : {"loaded from " + _TOX21_MODEL_DIR if _TOX21_AVAILABLE else "NOT configured (config.STAGE9_TOX21_MODEL_DIR) -- PAINS/Brenk panel only"}
""")

    pairs_by_arm, meta_by_arm, stats = build_masking_mode_pairs(
        max_molecules=max_molecules, sample_seed=sample_seed,
        mask_percent=mask_percent, match=match,
    )
    if not pairs_by_arm:
        print("  No usable Stage 1b rows -- nothing to compare.\n"
              "  Point config.STAGE9_9A_STAGE1B_DATA_DIR (or "
              "config.STAGE1B_PLIP_MASK_DIR) at the directory or .tar.gz that "
              "holds stage1b_large_scale_plip_mask_summary.csv, or run "
              "stage1b_large_scale_PLIP_mask_calculation.py first.")
        if stats.get("modes_seen"):
            print(f"  (masking_mode values seen: {stats['modes_seen']})")
        sys.exit(1)

    print(f"\n  {stats['molecules']} molecule(s) per arm "
          f"({stats.get('candidate_molecules', 0)} eligible before sampling).")
    for arm in ARM_ORDER:
        print(f"    {ARM_LABELS[arm]:<40} {len(pairs_by_arm.get(arm, []))} pairs")

    # ── the GPU pass: identical code for all three arms ───────────────────
    tokenizer, model, device = load_chemberta_base()
    records_by_arm = evaluate_property_records(
        tokenizer, model, device, pairs_by_arm, source_order=ARM_ORDER,
    )

    # ── the shared parent reference ───────────────────────────────────────
    # All three arms have the SAME parents, so this is ONE series, not three:
    # the chemistry the molecules started from, which is what tells you
    # whether an arm's QED of 0.42 is a gain or a loss.
    if parent_reference:
        one_arm = next(a for a in ARM_ORDER if pairs_by_arm.get(a))
        parent_records = evaluate_parent_property_records(
            {PARENT: pairs_by_arm[one_arm]})
        records_by_arm[PARENT] = parent_records.get(PARENT, [])

    os.makedirs(config.STAGE9A_DIR, exist_ok=True)
    fig_path = os.path.join(config.STAGE9A_DIR, FIG_NAME[match])
    csv_path = os.path.join(config.STAGE9A_DIR, CSV_NAME[match])

    print("\n  GENERATED molecules, by masking mode (vanilla ChemBERTa):")
    for line in format_property_summary(
        records_by_arm, source_order=ARM_ORDER, source_labels=ARM_LABELS,
    ):
        print(line)
    if records_by_arm.get(PARENT):
        # Validity and novelty are omitted for the reasons Stage 9a omits them
        # on its own parent arm: 100% valid by construction (unparseable
        # parents never became rows), and novelty 0 against itself.
        print("\n  PARENT (pre-mask) molecules -- the reference all three arms "
              "started from:")
        for line in format_property_summary(
            records_by_arm, omit=("validity", "novelty"),
            source_order=(PARENT,), source_labels=ARM_LABELS,
        ):
            print(line)

    plot_property_report(
        records_by_arm,
        out_path      = fig_path,
        # One line: plot_property_report puts the legend row directly under
        # the suptitle, so a second line lands behind it. The conditions the
        # second line would have stated are in the footer, where they belong.
        suptitle      = ("Stage 9a -- which part of the ligand you mask: PLIP "
                         "interacting vs non-interacting vs random "
                         "(vanilla ChemBERTa, no fine-tuning)"),
        footer        = format_comparison_footer(stats, meta_by_arm,
                                                 mask_percent, match),
        source_order  = [a for a in PLOT_ORDER if records_by_arm.get(a)],
        source_labels = ARM_LABELS,
        source_colors = ARM_COLORS,
        short_labels  = SHORT_LABELS,
        omit_panels   = OMIT_PANELS,
        # The parent is 100% RDKit-valid BY CONSTRUCTION (unparseable parents
        # never became rows), so a parent bar on the validity panel would
        # report the data filter, not the molecules.
        panel_series  = {"validity": [a for a in ARM_ORDER
                                      if records_by_arm.get(a)]},
        hatches       = {PARENT: PARENT_HATCH},
    )
    write_comparison_csv(records_by_arm, meta_by_arm, csv_path)

    print(f"\n  Figure : {fig_path}")
    print(f"  CSV    : {csv_path}\n")


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST  (no model, no GPU -- the arm construction is what can be wrong)
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    """
    Pins the three things that would silently invalidate the comparison:
    the interacting set being recovered wrongly from either storage mode, the
    arms drifting onto different molecules or different budgets, and an arm
    masking atoms it was not allowed to mask.
    """
    import tempfile

    print("\n" + "=" * 62)
    print("  STAGE 9a MASKING-MODE COMPARISON SELF-TEST")
    print("=" * 62)

    tokenizer = get_chemberta_tokenizer()

    # A molecule big enough that 15% of its tokens is a real budget, with a
    # plausible "interacting" subset.
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol    = Chem.MolFromSmiles(smiles)
    n      = mol.GetNumAtoms()
    interacting = [0, 1, 2, 3]
    non_interacting = sorted(set(range(n)) - set(interacting))

    def _rows(mode: str, stored: List[int]) -> List[dict]:
        return [{
            "pdb_id": "1abc", "resname": "LIG", "chain": "A", "resseq": "1",
            "masking_mode": mode, "status": "ok", "error": "",
            "smiles": smiles, "masked_smiles": "", "bpe_mask_count": "",
            "masked_atom_indices": json.dumps(stored),
        }]

    def _write(tmpdir: str, rows: List[dict]) -> str:
        path = os.path.join(tmpdir, "stage1b_large_scale_plip_mask_summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return tmpdir

    # [1] The SAME interacting set must come out of both storage modes: one
    #     stores it directly, the other stores its complement.
    built = {}
    for mode, stored in (("attractive", interacting),
                         ("non-attractive", non_interacting)):
        with tempfile.TemporaryDirectory() as td:
            _write(td, _rows(mode, stored))
            pairs, meta, stats = build_masking_mode_pairs(
                stage1b_dir=td, dedup_by_parent=False)
            assert stats["molecules"] == 1, (mode, stats)
            built[mode] = (pairs, meta)
    a_pairs, a_meta = built["attractive"]
    n_pairs, n_meta = built["non-attractive"]
    for arm in ARM_ORDER:
        assert a_pairs[arm] == n_pairs[arm], (
            f"{arm} differs between the two storage modes -- the complement "
            f"recovery is wrong")
    assert a_meta[INTERACTING][0]["n_interacting"] == len(interacting)
    assert a_meta[INTERACTING][0]["n_non_interacting"] == len(non_interacting)
    print("  [1] interacting set recovered identically from 'attractive' and "
          "'non-attractive' rows   OK")

    # [2] Same molecules in every arm, and the SAME NUMBER OF <mask> TOKENS --
    #     the quantity the default design matches, and the one the model
    #     actually sees. Counted in the built STRING, not in the bookkeeping,
    #     so the assertion cannot pass on a number the masker disagreed with.
    parents = {arm: [orig for _, orig in a_pairs[arm]] for arm in ARM_ORDER}
    assert len(set(map(tuple, parents.values()))) == 1, parents
    tokens = {arm: sorted(masked.count(tokenizer.mask_token)
                          for masked, _ in a_pairs[arm]) for arm in ARM_ORDER}
    assert len(set(map(tuple, tokens.values()))) == 1, tokens
    assert tokens[INTERACTING][0] >= 1, tokens
    recorded = {arm: sorted(m["n_mask_tokens"] for m in a_meta[arm])
                for arm in ARM_ORDER}
    assert recorded == tokens, (recorded, tokens)
    print(f"  [2] all three arms cover the same molecule(s) with the same "
          f"{tokens[INTERACTING][0]} <mask> token(s)   OK")

    # [3] Each arm masks only atoms from its own region. Checked through the
    #     masker itself rather than by re-deriving the sample: the claim is
    #     about what ends up in the STRING, which is what the model sees.
    for arm, forbidden in ((INTERACTING,     non_interacting),
                           (NON_INTERACTING, interacting)):
        masked = a_pairs[arm][0][0]
        # Masking the FORBIDDEN region to the same token count must give a
        # different string; if the arm drew from the wrong pool they collide.
        token_of = atom_token_map(smiles, tokenizer)
        assert token_of is not None
        picked   = _select_atoms_for_tokens(
            forbidden, token_of, tokens[arm][0], "1abc:LIG:A:1",
            config.STAGE9_MASK_SEED)
        if picked:
            other = mask_atoms_in_smiles_token_level(smiles, picked, tokenizer)
            assert masked != other, arm
        assert tokenizer.mask_token in masked
    print("  [3] each arm's masked string comes from its own region   OK")

    # [3b] --match atoms still works, and matches ATOMS instead.
    with tempfile.TemporaryDirectory() as td:
        _write(td, _rows("non-attractive", non_interacting))
        at_pairs, at_meta, at_stats = build_masking_mode_pairs(
            stage1b_dir=td, dedup_by_parent=False, match=MATCH_ATOMS)
        assert at_stats["molecules"] == 1, at_stats
        atom_counts = {arm: sorted(m["n_mask_atoms"] for m in at_meta[arm])
                       for arm in ARM_ORDER}
        assert len(set(map(tuple, atom_counts.values()))) == 1, atom_counts
    print(f"  [3b] --match atoms matches ATOMS instead "
          f"({atom_counts[INTERACTING][0]} per arm)   OK")

    # [4] A ligand PLIP resolved no contacts for cannot form arm 1, and is
    #     dropped from ALL arms rather than silently becoming a 2-arm row.
    with tempfile.TemporaryDirectory() as td:
        _write(td, _rows("non-attractive", list(range(n))))   # everything is
        pairs, meta, stats = build_masking_mode_pairs(        # non-interacting
            stage1b_dir=td, dedup_by_parent=False)
        assert stats["no_interacting_atoms"] == 1, stats
        assert not pairs, pairs
    print("  [4] a row with no interacting atoms drops from every arm   OK")

    # [5] Indices that don't address the stored molecule are refused, rather
    #     than producing a complement over the wrong atom count.
    with tempfile.TemporaryDirectory() as td:
        _write(td, _rows("non-attractive", [0, 1, n + 50]))
        pairs, meta, stats = build_masking_mode_pairs(
            stage1b_dir=td, dedup_by_parent=False)
        assert stats["pool_out_of_range"] == 1, stats
        assert not pairs, pairs
    print("  [5] out-of-range stored indices refused   OK")

    # [6] Dedup keeps one row per distinct parent, and the arms stay paired.
    with tempfile.TemporaryDirectory() as td:
        rows = _rows("non-attractive", non_interacting) * 4
        _write(td, rows)
        pairs, meta, stats = build_masking_mode_pairs(
            stage1b_dir=td, dedup_by_parent=True)
        assert stats["molecules"] == 1, stats
        assert stats["duplicate_parents"] == 3, stats
        assert all(len(pairs[a]) == 1 for a in ARM_ORDER), pairs
    print("  [6] duplicate ligand instances collapse to one molecule   OK")

    print("\nStage 9a masking-mode comparison self-test passed.")


def _parse_args(argv: list) -> tuple:
    limit, seed, parent, match = None, None, True, MATCH_TOKENS
    if "--match" in argv:
        raw = argv[argv.index("--match") + 1].lower()
        if raw not in (MATCH_TOKENS, MATCH_ATOMS):
            sys.exit(f"--match takes {MATCH_TOKENS!r} or {MATCH_ATOMS!r}, "
                     f"got {raw!r}")
        match = raw
    if "--limit" in argv:
        raw   = argv[argv.index("--limit") + 1]
        limit = None if raw.lower() in ("none", "all", "0") else int(raw)
    if "--seed" in argv:
        seed = int(argv[argv.index("--seed") + 1])
    if "--parent-reference" in argv:
        parent = argv[argv.index("--parent-reference") + 1].lower() not in (
            "off", "no", "false", "0")
    return limit, seed, parent, match


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        _limit, _seed, _parent, _match = _parse_args(sys.argv[1:])
        main(max_molecules=_limit if "--limit" in sys.argv else None,
             sample_seed=_seed, parent_reference=_parent, match=_match)

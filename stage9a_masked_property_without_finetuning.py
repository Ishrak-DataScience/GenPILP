# -*- coding: utf-8 -*-
"""
stage9a_masked_property_without_finetuning.py
================================================
Inference-time property profile of VANILLA ChemBERTa (no LoRA, no
fine-tuning) on the masked-molecule data produced by Stage 1a
(config.STAGE1A_DIR) and/or Stage 1b (config.STAGE1B_PLIP_MASK_DIR) --
plus, as the reference arm that comparison needs, the SAME property profile
measured on the PARENT (pre-mask) molecules themselves. Either source alone
is enough; a missing source is warned about and skipped, never a hard stop.

Why this script exists
-----------------------
stage9_masked_property_finetune.py REINFORCE-tunes ChemBERTa's LoRA adapters
against a six-term composite score. To see what fine-tuning actually bought,
we need the SAME criteria measured on the SAME molecules, decoded the SAME
way, with the frozen pretrained checkpoint and no adapter at all. That is
this script: every <mask> position is filled in a single forward pass
(config.CHEMBERTA_MODEL, one-shot MLM decoding via reinforce_rollout_oneshot
-- ChemBERTa's native "predict all masks at once" behaviour, not sequential
left-to-right filling) and the result is measured with Stage 9's own
compute_property_components, then plotted by Stage 9's own
plot_property_report. Both figures therefore come from identical code: the
only thing that differs is the model weights.

The parent arm answers the question one level below that. "Vanilla ChemBERTa
reaches QED 0.42" means nothing until you know what the molecules scored
BEFORE anything was masked and refilled. So the parents go through
evaluate_parent_property_records -- the same compute_property_components, no
model involved -- and land on their own figure and, overlaid with the
generated arm, on a direct before/after figure.

Why the parent figure has no validity or novelty panel
-------------------------------------------------------
  * RDKit validity  -- collect_pairs_from_stage1a/1b DROP every row whose
                       parent SMILES RDKit rejects (counted as
                       "invalid_parent" in the footer) before a pair is ever
                       built, so the scored parents are 100% valid BY
                       CONSTRUCTION. A 100% bar there would read as a
                       property of the chemistry rather than of the filter,
                       so the panel is omitted from the parent figure and,
                       on the overlay, replaced by a dotted reference line.
  * Scaffold novelty -- novelty is 1 - Tanimoto(parent, generated). A parent
                       scored against itself is 0 by definition, an identity
                       and not a measurement, so it is reported as "not
                       measurable" (None) and never plotted.
Everything else -- QED, synthetic accessibility, PAINS/Brenk structural
alerts, and the Tox21 classifier when config.STAGE9_TOX21_MODEL_DIR is set
-- is measured on parents exactly as on generated molecules.

How invalid molecules are counted (generated arm)
--------------------------------------------------
An RDKit-invalid generation is recorded, but only as a validity data point:

  * RDKit validity      -- rate over ALL generated molecules
                           (valid / total generated). Invalid molecules are
                           counted here and NOWHERE else.
  * QED, SA, novelty,
    PAINS/Brenk, Tox21  -- measured over the VALID subset only. An invalid
                           molecule contributes no value at all, instead of
                           a fake worst-case 0 that would drag every
                           distribution toward zero.

Novelty is additionally skipped when the parent SMILES itself cannot be
fingerprinted, and SA when rdkit.Contrib.SA_Score is unavailable -- "not
measurable" is never silently reported as 0.

Run modes
----------
The script asks at startup which arms to run, because the two cost wildly
different amounts: the generated arm is one MLM forward pass per molecule
(the entire runtime of this script), the parent arm is pure RDKit.

  3) both        (default)  score parents AND generate -- every figure below:
                            the generated arm, the all-parents arm, the
                            parents-of-valid arm, and the overlay.
  1) parents only           NO model weights, NO GPU pass. ChemBERTa's
                            TOKENIZER is still loaded (a few kB of
                            vocab/merges, no weights, no forward pass): Stage
                            1b pairs are re-masked to
                            config.STAGE9_MASK_PERCENT% of each molecule's BPE
                            tokens, so collect_pairs_from_stage1a/1b cannot
                            build a pair without it. Seeing HuggingFace fetch
                            config.json / vocab.json / merges.txt here is that
                            tokenizer, NOT the model. If a previous run's
                            stage9a_per_molecule.csv is on disk its generated
                            records are reloaded from it, so BOTH remaining
                            figures -- the parents-of-valid arm and the
                            overlay -- are still produced, against exactly the
                            molecules that run scored rather than merely the
                            same sample seed. Use this to add or restyle the
                            parent figures without paying for inference again.
                            Without that CSV only the all-parents figure can
                            be made: which parents survived is a property of a
                            generation run, so it cannot be recovered from the
                            Stage 1a/1b sources alone.
  2) generated only         exactly the pre-parent-arm behaviour.

A non-interactive session (stdin is not a TTY -- cron, nohup, a notebook)
never prompts and runs "both", the same convention
confirm_partial_sources_or_exit uses. --mode parent|generated|both overrides
the prompt entirely.

What gets written
-----------------
  <config.STAGE9A_DIR>/stage9a_property_distributions.png
      GENERATED arm. Six panels (five without a Tox21 checkpoint), each
      stating its own denominator: validity bar with a Wilson 95% CI; QED and
      raw-SA histograms with box strips; novelty box-over-violin with the
      "identical to parent" fraction called out (this distribution is
      bimodal -- an MLM often refills a mask with the parent's own atoms);
      PAINS / Brenk / any-alert hit rates over valid molecules. A footer
      states how many rows were read and what was dropped during data prep.

  <config.STAGE9A_DIR>/stage9a_parent_property_distributions.png
      ALL-PARENTS arm, same panels minus validity and novelty (see above).

  <config.STAGE9A_DIR>/stage9a_parent_of_valid_property_distributions.png
      PARENTS-OF-VALID arm: the same panels over only those parents whose
      masked child ChemBERTa refilled into an RDKit-parseable molecule (7,652
      of 41,474, in the run this was built for). Every panel on the generated
      figure except validity is already conditioned on the generation being
      valid, so this is the arm those panels should actually be read against
      -- it covers the identical molecules, which the all-parents figure does
      not. See select_parents_of_valid_children.

  <config.STAGE9A_DIR>/stage9a_parent_vs_generated.png
      Both arms overlaid panel by panel, parent and generated of the same
      source sharing a colour (they are the same molecules) and told apart by
      hatching. This is the figure to read for "what did masking and refilling
      with an untuned ChemBERTa do to the chemistry".

  <config.STAGE9A_DIR>/stage9a_per_molecule.csv
  <config.STAGE9A_DIR>/stage9a_parent_per_molecule.csv
  <config.STAGE9A_DIR>/stage9a_parent_of_valid_per_molecule.csv
      One row per attempt / per parent / per surviving parent (masked /
      parent / generated SMILES plus every measured property, blank where not
      measurable), so every number on every figure can be recomputed or
      re-plotted without another GPU pass -- which is exactly what "parents
      only" mode does with the first of them.

How many molecules
------------------
config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE caps how many pairs are scored PER
SOURCE (not in total, so neither panel is starved by the other source having
more rows), sampled with config.STAGE9_PAIR_SAMPLE_SEED BEFORE any molecule
is generated -- so a smaller limit really is a shorter run. Set it to None to
score everything. Stage 9's post-training pass samples identically, so the
before/after figures still cover the same molecules.

Usage
-----
  python stage9a_masked_property_without_finetuning.py
  python stage9a_masked_property_without_finetuning.py --mode parent
  python stage9a_masked_property_without_finetuning.py --mode generated
  python stage9a_masked_property_without_finetuning.py --limit 200
  python stage9a_masked_property_without_finetuning.py --limit none --seed 7
  python stage9a_masked_property_without_finetuning.py --test   (synthetic smoke test)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import torch
from rdkit import RDLogger
from transformers import AutoModelForMaskedLM

RDLogger.DisableLog("rdApp.*")

import config
from stage9_masked_property_finetune import (
    EVAL_MAX_PAIRS_PER_SOURCE,
    EVAL_DEDUP_BY_PARENT as DEDUP_BY_PARENT,
    MAX_MODEL_TOKENS,
    SAMPLE_SEED,
    SOURCE_COLORS,
    SOURCE_LABELS,
    TOP_K,
    TEMPERATURE,
    _SOURCE_ORDER,
    _TOX21_AVAILABLE,
    _TOX21_MODEL_DIR,
    collect_pairs_by_source,
    confirm_partial_sources_or_exit,
    evaluate_parent_property_records,
    evaluate_property_records,
    format_collection_footer,
    format_property_summary,
    get_chemberta_tokenizer,
    plot_property_report,
    read_property_records_csv,
    write_property_records_csv,
)


# ════════════════════════════════════════════════════════════════════════════
#  OUTPUT NAMES  (one place, because "parents only" mode reads one of them
#  back rather than regenerating it)
# ════════════════════════════════════════════════════════════════════════════

GENERATED_FIG_NAME    = "stage9a_property_distributions.png"
GENERATED_CSV_NAME    = "stage9a_per_molecule.csv"
PARENT_FIG_NAME       = "stage9a_parent_property_distributions.png"
PARENT_CSV_NAME       = "stage9a_parent_per_molecule.csv"
PARENT_VALID_FIG_NAME = "stage9a_parent_of_valid_property_distributions.png"
PARENT_VALID_CSV_NAME = "stage9a_parent_of_valid_per_molecule.csv"
OVERLAY_FIG_NAME      = "stage9a_parent_vs_generated.png"


# ════════════════════════════════════════════════════════════════════════════
#  PARENT SERIES  (plot_property_report draws any series it is given labels
#  and colours for -- see its source_order / source_labels / source_colors)
# ════════════════════════════════════════════════════════════════════════════

# A parent series shares its source's COLOUR deliberately: parent and
# generated of the same source are the same molecules before and after
# masking, and colouring them apart would suggest four independent datasets.
# The hatch is what separates them, so the overlay survives greyscale
# printing and the colourblind-safe palette keeps its two-hue budget.
PARENT_SUFFIX = "_parent"
PARENT_HATCH  = "///"
# Third arm: the SUBSET of parents whose masked child ChemBERTa refilled into
# something RDKit could parse. Same molecules, same measurements, smaller n --
# see select_parents_of_valid_children for why it is worth plotting apart.
PARENT_VALID_SUFFIX = "_parent_valid"

PARENT_LABELS = {
    f"stage1a{PARENT_SUFFIX}": "Parent, pre-mask (Stage 1a set)",
    f"stage1b{PARENT_SUFFIX}": "Parent, pre-mask (Stage 1b set)",
    f"stage1a{PARENT_VALID_SUFFIX}": "Parent of RDKit-valid generation (Stage 1a set)",
    f"stage1b{PARENT_VALID_SUFFIX}": "Parent of RDKit-valid generation (Stage 1b set)",
}
PARENT_COLORS = {
    **{f"{s}{PARENT_SUFFIX}":       SOURCE_COLORS[s] for s in _SOURCE_ORDER},
    **{f"{s}{PARENT_VALID_SUFFIX}": SOURCE_COLORS[s] for s in _SOURCE_ORDER},
}
# Stats-box names. The overlay carries four series, and four full legend
# labels per panel run off the edge of the panel -- the legend still spells
# them out, the numbers box uses these.
SHORT_LABELS  = {
    "stage1a":                       "Generated 1a",
    "stage1b":                       "Generated 1b",
    f"stage1a{PARENT_SUFFIX}":       "Parent 1a",
    f"stage1b{PARENT_SUFFIX}":       "Parent 1b",
    f"stage1a{PARENT_VALID_SUFFIX}": "Parent-of-valid 1a",
    f"stage1b{PARENT_VALID_SUFFIX}": "Parent-of-valid 1b",
}
PARENT_ORDER       = tuple(f"{s}{PARENT_SUFFIX}" for s in _SOURCE_ORDER)
PARENT_VALID_ORDER = tuple(f"{s}{PARENT_VALID_SUFFIX}" for s in _SOURCE_ORDER)
# Grouped by source, so the alert panel's bars sit parent-next-to-generated
# for one source before moving on to the next.
OVERLAY_ORDER = tuple(x for s in _SOURCE_ORDER for x in (f"{s}{PARENT_SUFFIX}", s))

# Parents are 100% RDKit-valid because unparseable ones were dropped in prep,
# not because of anything about the molecules -- so the overlay states it as a
# reference line instead of plotting bars that invite the wrong reading.
PARENT_VALIDITY_REFERENCE = (
    1.0, "parents, 100% valid by construction (unparseable ones dropped in prep)",
)


def as_parent_series(
    parent_records: Dict[str, List[Dict[str, object]]],
    suffix:         str = PARENT_SUFFIX,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Re-key parent records from "stage1a"/"stage1b" onto the "*_parent" series
    names (or "*_parent_valid" for the third arm), so a parent arm and the
    generated arm can live in ONE records dict and be drawn by one
    plot_property_report call rather than two overlaid figures.
    """
    return {f"{source}{suffix}": records
            for source, records in parent_records.items()}


def select_parents_of_valid_children(
    parent_records:    Dict[str, List[Dict[str, object]]],
    generated_records: Dict[str, List[Dict[str, object]]],
) -> Dict[str, List[Dict[str, object]]]:
    """
    The subset of parent records whose OWN masked child ChemBERTa refilled
    into a molecule RDKit could parse -- 7,652 of 41,474, in the run that
    motivated this arm.

    Why this is a separate distribution rather than a footnote: every panel
    on the generated figure except validity is already conditioned on the
    generation being valid, so its QED / SA / alert numbers describe a
    self-selected 18% of the input set. Comparing those against ALL 41,474
    parents silently compares two different populations, and any difference
    could be the model OR could be which molecules happen to survive masking
    (short, simple, low-SA molecules survive far more often than large
    natural-product-like ones). Plotting the parents of exactly the surviving
    children removes that confound: the generated panels and this figure then
    cover the identical molecules, so a shift between them is attributable to
    the masking-and-refilling, not to the sampling.

    Pairs by POSITION, which is exact: evaluate_parent_property_records and
    evaluate_property_records both walk the same pairs_by_source in the same
    order, and the cached path derives its pairs from the generated records
    themselves. Position also handles config.STAGE9_EVAL_DEDUP_BY_PARENT=False
    correctly, where one parent appears several times with different masks and
    only some of those attempts are valid.

    Falls back to matching on the parent SMILES -- with a warning, because the
    result is then approximate for the repeated-parent case -- if the two arms
    have drifted out of alignment (a hand-edited or stale cached CSV).
    """
    selected: Dict[str, List[Dict[str, object]]] = {}
    for source in _SOURCE_ORDER:
        par = parent_records.get(source) or []
        gen = generated_records.get(source) or []
        if not par or not gen:
            continue

        aligned = (len(par) == len(gen) and all(
            p.get("original_smiles") == g.get("original_smiles")
            for p, g in zip(par, gen)
        ))
        if aligned:
            kept = [p for p, g in zip(par, gen) if g.get("valid") == 1.0]
        else:
            valid_parents = {str(g.get("original_smiles"))
                             for g in gen if g.get("valid") == 1.0}
            kept = [p for p in par
                    if str(p.get("original_smiles")) in valid_parents]
            print(f"  WARNING: {SOURCE_LABELS.get(source, source)} parent and generated "
                  f"records are not positionally aligned ({len(par)} vs {len(gen)}); "
                  f"fell back to matching parents by SMILES, which is approximate "
                  f"when the same parent was masked more than once.")
        selected[source] = kept
    return selected


def format_valid_child_footer(
    parent_records:       Dict[str, List[Dict[str, object]]],
    parent_valid_records: Dict[str, List[Dict[str, object]]],
) -> str:
    """
    "N of M parents (P%) had an RDKit-valid generation", per source -- the
    figure's own denominator, stated on the figure rather than left to be
    looked up in the console log.
    """
    parts: List[str] = []
    for source in _SOURCE_ORDER:
        total = len(parent_records.get(source) or [])
        kept  = len(parent_valid_records.get(source) or [])
        if not total:
            continue
        parts.append(
            f"{SOURCE_LABELS.get(source, source)}: {kept} of {total} parent(s) "
            f"({kept / total:.1%}) had an RDKit-valid generation"
        )
    return "   |   ".join(parts)


# ════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (vanilla ChemBERTa -- no peft/LoRA import needed at all)
# ════════════════════════════════════════════════════════════════════════════

def load_chemberta_base(model_name: str = config.CHEMBERTA_MODEL):
    """
    Load ChemBERTa exactly as pretrained -- no LoRA adapter, no fine-tuning.
    This is the "before" baseline stage9_masked_property_finetune.py's LoRA
    adapter is compared against. The tokenizer is the same cached ChemBERTa
    BPE tokenizer the re-masking used, so mask counts and token limits agree
    end to end.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    tokenizer = get_chemberta_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


# ════════════════════════════════════════════════════════════════════════════
#  RUN MODE
# ════════════════════════════════════════════════════════════════════════════

RUN_MODES = ("both", "parent", "generated")

_MODE_MENU = """
  Which property profile(s) should this run produce?

    1) Parent molecules only    -- NO model weights are loaded and no molecule
                                   is generated: RDKit only, seconds not
                                   hours. (ChemBERTa's TOKENIZER is still
                                   fetched -- a few kB of vocab/merges files,
                                   no weights and no forward pass -- because
                                   re-masking Stage 1b to
                                   config.STAGE9_MASK_PERCENT% of each
                                   molecule's BPE tokens needs it to build the
                                   pairs at all.)
                                   A previous run's per-molecule CSV
                                   ({gen_csv})
                                   is reloaded if present, so the
                                   parents-of-valid figure and the
                                   parent-vs-generated overlay are still
                                   produced without re-running inference.
                                   Without it, only the all-parents figure can
                                   be made -- which parents survived is a
                                   property of a generation run.
    2) Generated molecules only -- fill the masks with vanilla ChemBERTa and
                                   plot those only (no parent reference).
    3) Both                     -- score the parents AND generate; all four
                                   figures (generated, all parents, parents of
                                   valid generations, overlay).
"""


def choose_run_mode(cli_mode: str = None) -> str:
    """
    Which arms to run: "both" (default), "parent", or "generated".

    --mode wins outright. Otherwise the user is asked, because the two arms
    differ by orders of magnitude in cost -- generating is one MLM forward
    pass per molecule and is the entire runtime of this script, while scoring
    parents is pure RDKit -- and because someone who already has
    stage9a_per_molecule.csv from an earlier run has no reason to pay for
    inference again just to add the parent panels.

    Auto-selects "both" without prompting when stdin isn't a TTY, so a
    scheduled or piped run can never hang -- same convention as
    confirm_partial_sources_or_exit and stage1_9's _ask_resume.
    """
    if cli_mode:
        if cli_mode not in RUN_MODES:
            raise SystemExit(f"--mode must be one of {', '.join(RUN_MODES)}, got {cli_mode!r}")
        print(f"  Run mode: {cli_mode} (from --mode).")
        return cli_mode

    if not sys.stdin.isatty():
        print("  Non-interactive session -- running BOTH arms (parents + generated).")
        return "both"

    print(_MODE_MENU.format(gen_csv=GENERATED_CSV_NAME))
    choices = {"1": "parent", "2": "generated", "3": "both", "": "both"}
    while True:
        ans = input("  Choice [3]: ").strip().lower()
        if ans in choices:
            return choices[ans]
        if ans in RUN_MODES:                     # typing the word works too
            return ans
        print("  Please enter 1, 2 or 3.")


def pairs_from_records(
    records_by_source: Dict[str, List[Dict[str, object]]],
) -> Dict[str, List[Tuple[str, str]]]:
    """
    (masked, original) pairs recovered from a per-molecule CSV that was read
    back with read_property_records_csv.

    Used by "parents only" mode: deriving the parent arm from the cached
    GENERATED records -- rather than re-running collect_pairs_by_source --
    guarantees the two arms cover the identical molecules in the identical
    order, instead of merely the same sample seed over a source CSV that may
    have been regenerated since. It also skips re-reading and re-masking the
    Stage 1b summary, which is the slow part of a no-GPU run.
    """
    return {
        source: [(str(r.get("masked_smiles") or ""), str(r.get("original_smiles") or ""))
                 for r in records if r.get("original_smiles")]
        for source, records in records_by_source.items()
        if source in _SOURCE_ORDER
    }


# ════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ════════════════════════════════════════════════════════════════════════════

def plot_parent_report(
    parent_records: Dict[str, List[Dict[str, object]]],
    out_path:       str,
    footer:         str = None,
) -> None:
    """
    The parent (pre-mask) molecules on their own, through the same
    plot_property_report every other figure in Stage 9/9a uses.

    Validity and novelty are omitted rather than drawn: see the module
    docstring -- one would report the collection filter, the other an
    identity.
    """
    plot_property_report(
        as_parent_series(parent_records), out_path,
        suptitle=("Stage 9a -- property profile of the PARENT (pre-mask) molecules "
                  "-- no model involved"),
        footer=footer,
        source_order=PARENT_ORDER,
        source_labels=PARENT_LABELS,
        source_colors=PARENT_COLORS,
        short_labels=SHORT_LABELS,
        omit_panels=("validity", "novelty"),
    )


def plot_parent_of_valid_report(
    parent_valid_records: Dict[str, List[Dict[str, object]]],
    out_path:             str,
    footer:               str = None,
) -> None:
    """
    The third arm: only those parents whose masked child came back RDKit-valid.

    Same three panels as the all-parents figure (QED, synthetic accessibility,
    structural alerts -- plus Tox21 when a checkpoint is configured), through
    the same plot_property_report, so it can be read directly against both the
    all-parents figure (does surviving masking select for particular
    chemistry?) and the generated figure (what did refilling do to the
    molecules that survived?). The second of those is the paired comparison:
    this figure and the generated panels cover the identical molecules.

    Validity and novelty are omitted for the same reasons as the all-parents
    figure -- one would report the collection filter, the other an identity.
    """
    plot_property_report(
        as_parent_series(parent_valid_records, PARENT_VALID_SUFFIX), out_path,
        suptitle=("Stage 9a -- property profile of the PARENTS whose masked child "
                  "ChemBERTa refilled into a VALID molecule"),
        footer=footer,
        source_order=PARENT_VALID_ORDER,
        source_labels=PARENT_LABELS,
        source_colors=PARENT_COLORS,
        short_labels=SHORT_LABELS,
        omit_panels=("validity", "novelty"),
    )


def plot_parent_vs_generated_report(
    parent_records:    Dict[str, List[Dict[str, object]]],
    generated_records: Dict[str, List[Dict[str, object]]],
    out_path:          str,
    footer:            str = None,
) -> None:
    """
    Parent and generated arms overlaid panel by panel -- the figure that
    actually answers "what did masking and refilling with an untuned
    ChemBERTa do to the chemistry".

    The validity and novelty panels stay generated-only (a parent bar there
    would report the collection filter and an identity respectively), with the
    parents' 100%-by-construction validity shown as a dotted reference line so
    nothing is silently missing.
    """
    combined = {**as_parent_series(parent_records), **generated_records}
    generated_only = [s for s in _SOURCE_ORDER if generated_records.get(s)]
    plot_property_report(
        combined, out_path,
        suptitle=("Stage 9a -- parent (pre-mask) vs. generated molecules, "
                  "vanilla ChemBERTa (no fine-tuning)"),
        footer=footer,
        source_order=OVERLAY_ORDER,
        source_labels=PARENT_LABELS,
        source_colors=PARENT_COLORS,
        short_labels=SHORT_LABELS,
        panel_series={"validity": generated_only, "novelty": generated_only},
        hatches={s: PARENT_HATCH for s in PARENT_ORDER},
        validity_reference=PARENT_VALIDITY_REFERENCE,
    )


def describe_arm_overlap(
    parent_records:    Dict[str, List[Dict[str, object]]],
    generated_records: Dict[str, List[Dict[str, object]]],
) -> str:
    """
    Whether the two arms cover the same molecules, as a line for the overlay's
    footer.

    They do whenever both arms come from one collect_pairs_by_source call
    ("both" mode) or the parents were derived from the cached generated
    records ("parents only" mode with a cache) -- which is every normal path.
    But a cached CSV written under a different --limit / --seed, or before the
    source data was regenerated, would silently turn a paired before/after
    comparison into two unrelated population distributions. That is worth
    stating on the figure rather than leaving the reader to assume pairing.
    """
    notes: List[str] = []
    for source in _SOURCE_ORDER:
        par = parent_records.get(source) or []
        gen = generated_records.get(source) or []
        if not par or not gen:
            continue
        par_set = {str(r.get("original_smiles")) for r in par}
        gen_set = {str(r.get("original_smiles")) for r in gen}
        shared  = len(par_set & gen_set)
        if len(par) == len(gen) and par_set == gen_set:
            notes.append(f"{SOURCE_LABELS.get(source, source)}: paired, same {len(par)} molecule(s)")
        else:
            notes.append(
                f"{SOURCE_LABELS.get(source, source)}: NOT PAIRED -- {len(par)} parent(s) vs "
                f"{len(gen)} generated, {shared} parent molecule(s) in common; the two arms "
                f"are population distributions here, not a per-molecule before/after"
            )
    return "   |   ".join(notes)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(
    max_pairs_per_source: int = None,
    sample_seed:          int = None,
    mode:                 str = None,
) -> None:
    """
    `max_pairs_per_source` / `sample_seed` default to
    config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE and config.STAGE9_PAIR_SAMPLE_SEED;
    --limit / --seed override them for one run. `mode` is "both" (default),
    "parent" or "generated" -- see choose_run_mode.
    """
    if max_pairs_per_source is None:
        max_pairs_per_source = EVAL_MAX_PAIRS_PER_SOURCE
    if sample_seed is None:
        sample_seed = SAMPLE_SEED
    unit_note  = ("unique parent molecule" if DEDUP_BY_PARENT
                  else "ligand-instance pair")
    limit_note = (f"a random sample of {max_pairs_per_source} {unit_note}(s) PER "
                  f"SOURCE (seed {sample_seed})" if max_pairs_per_source
                  else f"every available {unit_note} (no limit)")

    print("\n" + "=" * 60)
    print("STAGE 9a -- BASELINE PROPERTY EVALUATION (ChemBERTa, NO fine-tuning)")
    print("=" * 60)
    print(f"""
  What this script does
  ----------------------
  1. Loads the SAME masked/original SMILES pairs Stage 9 trains and
     evaluates on, split by source: Stage 1a random-token masking
     ({config.STAGE9_9A_STAGE1A_DATA_DIR or config.STAGE1A_DIR}) and/or Stage 1b
     PLIP interaction masking
     ({config.STAGE9_9A_STAGE1B_DATA_DIR or config.STAGE1B_PLIP_MASK_DIR}),
     re-masked to config.STAGE9_MASK_PERCENT={config.STAGE9_MASK_PERCENT}%
     of each molecule's BPE tokens -- whichever source(s) are available,
     either one alone is enough. Rows whose parent SMILES RDKit rejects, or
     that exceed ChemBERTa's {MAX_MODEL_TOKENS}-token window, are dropped
     and counted rather than crashing the run.
     {"Stage 1b's one-row-per-ligand-instance rows are collapsed to ONE pair per unique parent molecule first, so a ligand solved in many PDB entries is scored once rather than once per structure." if DEDUP_BY_PARENT else "Every ligand instance is scored separately (config.STAGE9_EVAL_DEDUP_BY_PARENT=False), so a ligand solved in many PDB entries is scored once per structure."}
     Scores {limit_note}
     -- config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE and
     config.STAGE9_PAIR_SAMPLE_SEED, overridable per run with --limit /
     --seed. Stage 9's post-training pass draws the SAME sample, so the
     before/after figures still cover the same molecules.
  2. PARENT ARM (no model): measures every pre-mask parent molecule on QED,
     synthetic accessibility, PAINS/Brenk structural alerts and -- if
     configured -- Tox21, giving the reference the generated numbers are read
     against. Validity and novelty are omitted for parents on purpose: the
     first is 100% by construction (unparseable parents never became pairs),
     the second is 0 by definition against itself.
  3. GENERATED ARM: loads vanilla ChemBERTa ({config.CHEMBERTA_MODEL}) -- NO
     LoRA adapter, NO fine-tuning, frozen pretrained weights only -- fills
     every <mask> in ONE forward pass per pair (one-shot MLM decoding, top-{TOP_K}
     sampling at temperature {TEMPERATURE}, the same reinforce_rollout_oneshot
     Stage 9's training rollout uses) and measures the result on everything
     Stage 9 optimises: RDKit validity, QED, synthetic accessibility, scaffold
     novelty vs. the parent (1 - Tanimoto), PAINS/Brenk structural alerts, and
     -- if configured -- a Tox21 classifier probability.
  4. Counts an INVALID generation for validity only (valid / all generated);
     it contributes nothing to QED / SA / novelty / alert statistics, which
     are computed over the valid subset with their own denominators printed
     on every panel.
  5. Writes four figures -- generated, ALL parents, the parents whose masked
     child came back RDKit-VALID (the paired reference for every non-validity
     panel of the generated figure), and the parent-vs-generated overlay --
     plus a per-molecule CSV per arm into
     {config.STAGE9A_DIR}

  Tox21 classifier : {"loaded from " + _TOX21_MODEL_DIR if _TOX21_AVAILABLE else "NOT configured (config.STAGE9_TOX21_MODEL_DIR) -- only PAINS/Brenk toxicity-risk panel will be plotted"}
""")

    mode = choose_run_mode(mode)

    os.makedirs(config.STAGE9A_DIR, exist_ok=True)
    gen_fig_path = os.path.join(config.STAGE9A_DIR, GENERATED_FIG_NAME)
    gen_csv_path = os.path.join(config.STAGE9A_DIR, GENERATED_CSV_NAME)
    par_fig_path = os.path.join(config.STAGE9A_DIR, PARENT_FIG_NAME)
    par_csv_path = os.path.join(config.STAGE9A_DIR, PARENT_CSV_NAME)
    pav_fig_path = os.path.join(config.STAGE9A_DIR, PARENT_VALID_FIG_NAME)
    pav_csv_path = os.path.join(config.STAGE9A_DIR, PARENT_VALID_CSV_NAME)
    ovl_fig_path = os.path.join(config.STAGE9A_DIR, OVERLAY_FIG_NAME)

    generated_records:    Dict[str, List[Dict[str, object]]] = {}
    parent_records:       Dict[str, List[Dict[str, object]]] = {}
    parent_valid_records: Dict[str, List[Dict[str, object]]] = {}
    cached_generated  = False
    footer            = None

    # ── Where the molecules come from ──────────────────────────────────────
    # "parents only" prefers a previous run's per-molecule CSV: it pins the
    # parent arm to exactly the molecules that run generated on (so the
    # overlay is a true per-molecule before/after), and it avoids re-reading
    # and re-masking the Stage 1b summary for a run that isn't generating.
    if mode == "parent":
        cached = read_property_records_csv(gen_csv_path)
        # A cache with no rows under a KNOWN source (a truncated or
        # hand-edited CSV) is treated as no cache at all, rather than being
        # allowed to fall through to the "no masked data found" exit below
        # and report a data-location problem that isn't one.
        cached = {s: r for s, r in cached.items() if s in _SOURCE_ORDER and r}
        if cached:
            generated_records = cached
            cached_generated  = True
            pairs_by_source   = pairs_from_records(generated_records)
            footer = (f"molecules taken from the cached per-molecule records in "
                      f"{GENERATED_CSV_NAME} (no inference run); "
                      f"see that run's figure footer for what data prep dropped")
            print(f"  Parent arm will cover the {sum(len(p) for p in pairs_by_source.values())} "
                  f"molecule(s) the cached run generated on.")
        else:
            print("  No cached generated records -- collecting parent molecules from "
                  "the Stage 1a/1b sources instead (parent figure only, no overlay).")
            pairs_by_source = collect_pairs_by_source(
                max_pairs_per_source=max_pairs_per_source, sample_seed=sample_seed,
            )
    else:
        pairs_by_source = collect_pairs_by_source(
            max_pairs_per_source=max_pairs_per_source, sample_seed=sample_seed,
        )

    if not pairs_by_source:
        print("  NEITHER Stage 1a nor Stage 1b masked-data pairs were found -- nothing to "
              "evaluate. Either source alone is enough to run this script, so point "
              "config.STAGE9_9A_STAGE1A_DATA_DIR / config.STAGE9_9A_STAGE1B_DATA_DIR at "
              "wherever that stage's output (directory or .tar/.tar.gz) actually lives, "
              "or run stage1a_random_masking.py / "
              "stage1b_large_scale_PLIP_mask_calculation.py first.")
        sys.exit(1)

    for source, pairs in pairs_by_source.items():
        print(f"  {source}: {len(pairs)} pairs")
    if not cached_generated:
        confirm_partial_sources_or_exit(pairs_by_source)
        footer = format_collection_footer(list(pairs_by_source.keys()))

    # ── Parent arm (RDKit only) ────────────────────────────────────────────
    if mode in ("both", "parent"):
        parent_records = evaluate_parent_property_records(pairs_by_source)

    # ── Generated arm (the GPU pass) ───────────────────────────────────────
    if mode in ("both", "generated"):
        tokenizer, model, device = load_chemberta_base()
        generated_records = evaluate_property_records(
            tokenizer, model, device, pairs_by_source,
        )

    # ── Figures + CSVs ─────────────────────────────────────────────────────
    written: List[Tuple[str, str]] = []

    if generated_records and not cached_generated:
        plot_property_report(
            generated_records, gen_fig_path,
            suptitle=("Stage 9a -- inference-time property profile "
                      "(vanilla ChemBERTa, no fine-tuning)"),
            footer=footer,
        )
        write_property_records_csv(generated_records, gen_csv_path)
        written += [("Generated figure  ", gen_fig_path),
                    ("Generated CSV     ", gen_csv_path)]
    elif cached_generated:
        # Deliberately NOT re-plotted: the cached run already wrote this
        # figure from these exact records, and redrawing it would only risk
        # overwriting it with a differently-configured copy.
        print(f"  Reusing the cached generated figure at {gen_fig_path} (not re-plotted).")

    if parent_records:
        plot_parent_report(parent_records, par_fig_path, footer=footer)
        write_property_records_csv(parent_records, par_csv_path)
        written += [("Parent figure     ", par_fig_path),
                    ("Parent CSV        ", par_csv_path)]

    # Third arm -- only the parents whose child survived RDKit. Needs both
    # arms, so it is produced by "both" AND by "parents only" when that mode
    # found a cached generated CSV to select against.
    if parent_records and generated_records:
        parent_valid_records = select_parents_of_valid_children(
            parent_records, generated_records,
        )
        valid_footer = format_valid_child_footer(parent_records, parent_valid_records)
        print(f"  {valid_footer}")
        if any(parent_valid_records.values()):
            plot_parent_of_valid_report(
                parent_valid_records, pav_fig_path,
                footer="   |   ".join(x for x in (valid_footer, footer) if x),
            )
            write_property_records_csv(parent_valid_records, pav_csv_path)
            written += [("Parent-of-valid fig", pav_fig_path),
                        ("Parent-of-valid CSV", pav_csv_path)]
        else:
            print("  No generation was RDKit-valid in any source -- the "
                  "parents-of-valid-generations figure has nothing to plot "
                  "and was skipped.")

    if parent_records and generated_records:
        overlap = describe_arm_overlap(parent_records, generated_records)
        if "NOT PAIRED" in overlap:
            print(f"  WARNING: {overlap}")
        plot_parent_vs_generated_report(
            parent_records, generated_records, ovl_fig_path,
            footer="   |   ".join(x for x in (overlap, footer) if x),
        )
        written.append(("Overlay figure    ", ovl_fig_path))

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Stage 9a baseline evaluation complete.")
    if parent_records:
        print("\n  PARENT (pre-mask) molecules -- validity and novelty omitted "
              "(100% by construction / 0 by definition):")
        for line in format_property_summary(
            parent_records, omit=("validity", "novelty"),
        ):
            print(line)
    if any(parent_valid_records.values()):
        print("\n  PARENTS whose masked child came back RDKit-VALID -- the paired "
              "reference for the generated numbers below:")
        for line in format_property_summary(
            parent_valid_records, omit=("validity", "novelty"),
        ):
            print(line)
    if generated_records:
        print(f"\n  GENERATED molecules (vanilla ChemBERTa"
              f"{', reloaded from cache' if cached_generated else ''}):")
        for line in format_property_summary(generated_records):
            print(line)
    print()
    for label, path in written:
        print(f"  {label}: {path}")
    print("  Compare against stage9_property_distributions.png (same molecules, "
          "same code, fine-tuned weights).")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    """
    Synthetic smoke test: hand-built masked/original pairs across both
    sources, one completion each, report pipeline exercised end-to-end
    (including the all-invalid and single-source edge cases, the parent arm,
    the overlay, and the CSV round-trip "parents only" mode depends on).
    Requires network access (HuggingFace hub) and the
    torch/transformers/rdkit dependencies, same as every other
    ChemBERTa-using stage in this pipeline.
    """
    import csv
    import tempfile

    from stage9_masked_property_finetune import (
        compute_property_components,
        summarize_property_records,
    )

    # Invalid generation: validity is measured (0.0), everything else is
    # unmeasurable (None) -- never a fake zero.
    comps_bad = compute_property_components("not_a_smiles(((", "CCO")
    assert comps_bad["valid"] == 0.0
    assert all(comps_bad[k] is None for k in
               ("qed", "sa_raw", "novelty", "any_alert", "tox21"))

    comps_ok = compute_property_components("CCO", "CCO")
    assert comps_ok["valid"] == 1.0
    assert comps_ok["qed"] is not None
    assert comps_ok["novelty"] == 0.0, "identical molecules must score zero novelty"

    # Novelty against an unparseable PARENT is unknown, not zero.
    comps_bad_parent = compute_property_components("CCO", "n1c([PH]2(=O)(O)O)c1")
    assert comps_bad_parent["valid"] == 1.0
    assert comps_bad_parent["novelty"] is None

    # Denominator split: 1 of 2 valid -> validity 50%, QED n=1.
    stats = summarize_property_records([
        dict(valid=1.0, qed=0.6, sa_raw=3.0, sa_norm=0.7, novelty=0.4,
             pains=0.0, brenk=1.0, any_alert=1.0, n_alerts=2.0, tox21=None),
        dict(valid=0.0, qed=None, sa_raw=None, sa_norm=None, novelty=None,
             pains=None, brenk=None, any_alert=None, n_alerts=None, tox21=None),
    ])
    assert stats["n_attempted"] == 2 and stats["n_valid"] == 1
    assert abs(stats["validity"] - 0.5) < 1e-9
    assert stats["qed_n"] == 1 and abs(stats["qed_mean"] - 0.6) < 1e-9
    assert abs(stats["any_alert_rate"] - 1.0) < 1e-9

    test_pairs_by_source = {
        "stage1a": [
            ("CC(=O)OC1=CC=CC=C1C(=O)<mask>", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)<mask>", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ],
        "stage1b": [
            ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)<mask>", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ],
    }

    # ── Parent arm: no model, novelty deliberately unmeasurable ────────────
    parent_records = evaluate_parent_property_records(test_pairs_by_source)
    assert set(parent_records) == {"stage1a", "stage1b"}
    assert len(parent_records["stage1a"]) == 2 and len(parent_records["stage1b"]) == 1
    for recs in parent_records.values():
        for r in recs:
            assert r["valid"] == 1.0, "collected parents are valid by construction"
            assert r["qed"] is not None
            assert r["novelty"] is None, "a parent vs. itself is an identity, not a measurement"
            assert r["generated_smiles"] == ""
    par_stats = summarize_property_records(parent_records["stage1a"])
    assert par_stats["novelty_n"] == 0
    assert par_stats["qed_n"] == 2

    par_summary = format_property_summary(parent_records, omit=("validity", "novelty"))
    assert not any("RDKit validity" in line for line in par_summary)
    assert not any("Novelty" in line for line in par_summary)
    assert any("QED" in line for line in par_summary)

    tokenizer, model, device = load_chemberta_base()
    records = evaluate_property_records(
        tokenizer, model, device, test_pairs_by_source, top_k=5,
    )
    assert set(records.keys()) == {"stage1a", "stage1b"}
    assert len(records["stage1a"]) == 2
    assert len(records["stage1b"]) == 1
    assert all(r["valid"] in (0.0, 1.0) for recs in records.values() for r in recs)

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "test_stage9a_plot.png")
        plot_property_report(records, out_path, "self-test",
                             footer=format_collection_footer())
        assert os.path.isfile(out_path)

        # Single-source case must still produce a figure (no PLIP data).
        single_source_path = os.path.join(td, "test_stage9a_single.png")
        plot_property_report({"stage1a": records["stage1a"]},
                             single_source_path, "self-test single source")
        assert os.path.isfile(single_source_path)

        # All-invalid case: validity panel at 0%, no other panel populated.
        invalid_record: dict = dict(
            source="stage1a", masked_smiles="<mask>", original_smiles="CCO",
            generated_smiles="((", **compute_property_components("((", "CCO"),
        )
        all_invalid = {"stage1a": [invalid_record]}
        invalid_path = os.path.join(td, "test_stage9a_all_invalid.png")
        plot_property_report(all_invalid, invalid_path, "self-test all invalid")
        assert os.path.isfile(invalid_path)

        # Parent figure: four panels at most, and never validity or novelty.
        parent_path = os.path.join(td, "test_stage9a_parent.png")
        plot_parent_report(parent_records, parent_path, footer="self-test parent")
        assert os.path.isfile(parent_path)

        # Third arm: parents of the RDKit-valid generations only. Selection is
        # positional, so it must track WHICH attempt was valid, not just how
        # many were.
        pav = select_parents_of_valid_children(parent_records, records)
        for source in ("stage1a", "stage1b"):
            expect = [p for p, g in zip(parent_records[source], records[source])
                      if g["valid"] == 1.0]
            assert pav[source] == expect
            assert len(pav[source]) <= len(parent_records[source])
            assert all(r["novelty"] is None for r in pav[source])

        # A parent whose child was INVALID must not appear.
        one_bad = {"stage1a": [dict(records["stage1a"][0], valid=0.0)]}
        one_par = {"stage1a": [parent_records["stage1a"][0]]}
        assert select_parents_of_valid_children(one_par, one_bad)["stage1a"] == []
        one_ok = {"stage1a": [dict(records["stage1a"][0], valid=1.0)]}
        assert select_parents_of_valid_children(one_par, one_ok)["stage1a"] == one_par["stage1a"]

        # Misaligned arms fall back to SMILES matching rather than crashing.
        misaligned = select_parents_of_valid_children(
            parent_records, {"stage1a": records["stage1a"][:1]},
        )
        assert set(misaligned) <= {"stage1a"}

        valid_footer = format_valid_child_footer(parent_records, pav)
        assert "had an RDKit-valid generation" in valid_footer

        if any(pav.values()):
            pav_path = os.path.join(td, "test_stage9a_parent_of_valid.png")
            plot_parent_of_valid_report(pav, pav_path, footer=valid_footer)
            assert os.path.isfile(pav_path)

        # Overlay: four series in one figure.
        overlay_path = os.path.join(td, "test_stage9a_overlay.png")
        plot_parent_vs_generated_report(
            parent_records, records, overlay_path,
            footer=describe_arm_overlap(parent_records, records),
        )
        assert os.path.isfile(overlay_path)

        # Same molecules on both arms -> the overlay footer must say "paired".
        overlap = describe_arm_overlap(parent_records, records)
        assert "paired" in overlap and "NOT PAIRED" not in overlap

        # Mismatched arms must be called out, not silently overlaid.
        mismatched = describe_arm_overlap(
            parent_records, {"stage1a": records["stage1a"][:1], "stage1b": records["stage1b"]},
        )
        assert "NOT PAIRED" in mismatched

        csv_path = write_property_records_csv(records, os.path.join(td, "test_records.csv"))
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert {"valid", "qed", "sa_raw", "novelty", "any_alert"} <= set(rows[0])

        # CSV round-trip -- what "parents only" mode reloads. Empty cells must
        # come back as None (not measurable), never as 0.0.
        reloaded = read_property_records_csv(csv_path)
        assert set(reloaded) == {"stage1a", "stage1b"}
        assert sum(len(v) for v in reloaded.values()) == 3
        for source in ("stage1a", "stage1b"):
            for orig, back in zip(records[source], reloaded[source]):
                assert orig["original_smiles"] == back["original_smiles"]
                assert (orig["valid"] or 0.0) == back["valid"]
                for key in ("qed", "sa_raw", "novelty"):
                    if orig[key] is None:
                        assert back[key] is None, f"{key} must stay unmeasurable, not become 0"
                    else:
                        assert abs(orig[key] - back[key]) < 1e-6
        assert read_property_records_csv(os.path.join(td, "does_not_exist.csv")) == {}

        # Pairs recovered from the cache reproduce what was scored.
        recovered = pairs_from_records(reloaded)
        assert recovered["stage1a"] == test_pairs_by_source["stage1a"]
        assert recovered["stage1b"] == test_pairs_by_source["stage1b"]

        summary = format_property_summary(records)
        assert any("RDKit validity" in line for line in summary)

    print("Stage 9a self-test passed.")


def _parse_args(argv: list) -> tuple:
    """
    --limit N : pairs to score PER SOURCE this run, overriding
                config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE. "none"/"all"/0 = no
                limit (score every available pair).
    --seed  N : sampling seed, overriding config.STAGE9_PAIR_SAMPLE_SEED.
    --mode  M : "both" | "parent" | "generated", skipping the startup prompt.
    """
    limit = seed = mode = None   # None = "not given", fall back to config/prompt
    for flag in ("--limit", "--seed", "--mode"):
        if flag not in argv:
            continue
        idx = argv.index(flag)
        if idx + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value, e.g. {flag} "
                             f"{'parent' if flag == '--mode' else '200'}")
        raw = argv[idx + 1]
        if flag == "--limit":
            # 0 (not None) means "explicitly no limit" -- None would be
            # indistinguishable from the flag being absent and would silently
            # fall back to config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE.
            limit = 0 if raw.lower() in ("none", "all", "0") else int(raw)
        elif flag == "--seed":
            seed = int(raw)
        else:
            mode = raw.lower()
    return limit, seed, mode


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        _limit, _seed, _mode = _parse_args(sys.argv)
        main(max_pairs_per_source=_limit, sample_seed=_seed, mode=_mode)

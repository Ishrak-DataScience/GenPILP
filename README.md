# GenPLIP — protein–ligand-interaction-guided molecular generation with ChemBERTa

Take a crystal structure of a protein with a ligand bound, ask PLIP which ligand
atoms actually touch the protein, mask those atoms (or deliberately mask the
others), and let a masked-language model propose replacements. Then measure
whether the proposals are any good: valid, diverse, drug-like, non-toxic, and
better docked than the ligand you started from.

The repository contains three things:

1. **The generation pipeline** (Stages 1–8) — interaction-aware masking,
   ChemBERTa infilling, similarity and novelty analysis, GNINA docking.
2. **Property-guided fine-tuning** (Stages 9–10) — two ways to train the model
   towards a non-differentiable chemistry score, plus the infrastructure that
   makes those runs resumable, held-out-evaluated and reproducible.
3. **A baseline** (`baseline_analysis/`) — the untuned model measured under
   three masking strategies on 24 curated complexes, which is the reference
   every fine-tuning result is supposed to beat.

---

## Contents

- [Scientific premise](#scientific-premise)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Input data](#input-data)
- [Stage reference](#stage-reference)
- [Property-guided fine-tuning](#property-guided-fine-tuning)
- [Baseline analysis](#baseline-analysis)
- [Self-tests](#self-tests)
- [Hardware and HPC](#hardware-and-hpc)
- [Repository map](#repository-map)
- [Known limitations](#known-limitations)

---

## Scientific premise

ChemBERTa is a BERT-style masked language model trained on SMILES. Masking a few
tokens and asking it to refill them is exactly its pretraining task, so it needs
no architectural change to act as a molecular generator — the design question is
*which* tokens to mask.

This project masks by **protein–ligand interaction**. PLIP reports, per binding
site, which ligand atoms make hydrogen bonds, hydrophobic contacts, salt
bridges, π-stacking, π-cation, halogen and water-bridge interactions. Those
atoms can be masked (**PLIP ++**, "regenerate the part that binds") or preserved
while everything else is masked (**PLIP −−**, "keep the pharmacophore, vary the
scaffold"). Uniform random masking is the control.

Whether interaction-guided masking beats random masking is the question the
pipeline exists to answer, and it is **not** settled by construction — see
[Known limitations](#known-limitations).

---

## Quick start

```bash
git clone https://github.com/Ishrak-DataScience/GenPILP.git GenPLIP
cd GenPLIP
pip install -r requirements.txt        # or: uv sync

# tell the pipeline which machine it is on (see Configuration)
python -c "import config; print(config.CONFIG_PLATFORM, config.BASE_DIR)"

# smallest useful thing: mask 5 curated ligands and look at the masks
python stage1_mask_calculation.py

# the untuned-model baseline, end to end (see baseline_analysis/README.md)
python baseline_analysis/select_samples.py
python baseline_analysis/build_masks.py
```

Most scripts take `--test` and run a self-contained self-test with no GPU, no
cluster and no network.

---

## Installation

Python **3.11+** (`.python-version` pins 3.11).

```bash
pip install -r requirements.txt
```

| package | why |
|---|---|
| `torch`, `transformers` | ChemBERTa inference and fine-tuning |
| `rdkit` | SMILES parsing, QED, SA score, fingerprints, PAINS/Brenk alerts |
| `selfies` | SELFIES representation used by Stage 1's masking |
| `peft` | LoRA adapters (Stage 9 only) |
| `pandas`, `numpy`, `scipy`, `matplotlib`, `tqdm` | data handling, statistics, figures |
| `chembl-webresource-client` | Stage 0a / Stage 5 ChEMBL queries |

Two external binaries are **not** pip-installable:

- **OpenBabel** — used by Stage 1 to convert an extracted ligand PDB to SDF.
  RDKit is tried as a fallback, so Stage 1 works without it on most ligands.
  `apt-get install openbabel` / `conda install -c conda-forge openbabel`.
- **GNINA** — docking (Stages 6 and the baseline). Linux x86-64 only.
  Auto-downloaded to `config.GNINA_BINARY` on first use, or fetched explicitly
  with `bash baseline_analysis/get_gnina.sh`.

> **GNINA is v1.3.3** (`gnina.cuda12.8.static`, **≈2.1 GB**). It needs a driver
> new enough for CUDA 12.8; an older-CUDA build (v1.3.2, 1.4 GB) is the
> documented fallback. **v1.3 retrained the CNN scoring functions on
> CrossDock2020 v1.3**, so `CNNaffinity`/`CNNscore` from the v1.0.3 this repo
> used before 2026-09-17 are a different quantity — earlier docking results are
> superseded, not mergeable.

---

## Configuration

`config.py` **is not the configuration.** It is a selector: every script does
`import config`, and that module picks one `config_<platform>.py` and *becomes*
it.

Selection rule:

1. `$GENPLIP_CONFIG` wins outright if set (platform name, module name, filename
   or path). This is also what parent processes hand to worker processes.
2. Otherwise, every `config_*.py` that declares `CONFIG_PLATFORM` is a
   candidate, and the one whose `BASE_DIR` **exists as a directory on this
   machine** wins. A Colab config points at `/content/drive/...`, an HPC config
   at `/home/<user>/...`; only one is ever real on a given box.
3. If several match, the longest `BASE_DIR` wins and a warning names the others.
4. If none match, that is a hard error, not a silent fallback.

Shipped configs: `config_colab.py`, `config_HPC_jupyter.py`, `config_laptop.py`
— **243 keys each, identical except for paths and deliberate per-platform
hardware values**. A knob added to one must be added to all three, at the same
position; otherwise a script's `getattr` default silently applies on the other
machines and looks like a code bug. `config_high.py` is legacy and is ignored
(no `CONFIG_PLATFORM`).

Knobs worth knowing:

| key | default | meaning |
|---|---|---|
| `BASE_DIR`, `USER_PREFIX`, `EXPERIMENT_TAG` | per machine | everything else is derived from these |
| `CHEMBERTA_MODEL` | `seyonec/ChemBERTa-zinc-base-v1` | the model masked and fine-tuned throughout |
| `MASK_PERCENT` | `15` | fraction of BPE tokens masked |
| `INCLUDE_TYPES` | 8 PLIP types | which interactions count as "contacting" |
| `PLIP_LARGE_SCALE_PDB_ROOT` / `_XML_ROOT` | group filesystem | PDB mirror and pre-computed PLIP XML |
| `STAGE1B_PLIP_NEGATIVE_MASK_DIR` | group filesystem | the PLIP −− mask corpus |
| `GNINA_BINARY`, `GNINA_DOWNLOAD_URL` | v1.3.3 | docking binary |
| `STAGE10_SPLIT_MODE`, `STAGE10_VAL_FRAC` | `scaffold`, `0.10` | held-out partition |
| `BASELINE_*` | see `baseline_analysis/` | the untuned-model baseline |

---

## Input data

**PDB structures.** Either a local mirror in the PDB "divided" layout
(`<root>/<mid2>/pdb<id>.ent.gz`, `mid2 = id[1:3]`) or individual files. For a
handful of complexes, `baseline_analysis/fetch_structures.py` downloads them
from RCSB instead.

**PLIP XML.** Pre-computed interaction reports, one per structure
(`<root>/pdb<id>.xml`). Produced by [PLIP](https://github.com/pharmai/plip); the
pipeline never calls PLIP itself, it only parses the XML.

**`metadata.tsv`** (tracked here, 109,597 rows) — one row per binding site:
`PDB:LIG:CHAIN:POS`, UniProt ID, ligand code, FDA status (`-1` special, `0` not
in trials, `1`/`2`/`3` trial phase, `4` approved) and per-type interaction
counts. This is what `baseline_analysis/select_samples.py` selects from.

**ChEMBL** (Stages 0a, 5) — downloaded via `chembl-webresource-client`.

---

## Stage reference

| stage | script | in | out |
|---|---|---|---|
| 0a | `stage0a_chembl_2M_download.py` | ChEMBL API | RDKit-verified SMILES file |
| 1 | `stage1_mask_calculation.py` | PDB + PLIP XML | masked SELFIES/SMILES, `.meta.json`, 2D interaction plots |
| 1a | `stage1a_random_masking*.py` | Stage 1 / 1b output | random-masking control at several mask rates |
| 1b | `stage1b_large_scale_PLIP_mask_calculation.py` | PDB + PLIP mirrors | the mask corpus (one summary CSV, hundreds of thousands of sites) |
| 1c | `stage1c_upload_pdb_plip_to_gdrive.py` | local pairs | Google Drive upload |
| 2 | `stage2_molecule_generation.py` | masked strings | generated SMILES |
| 2.5 / 2.7 | `stage2_5_random_pick_generation.py`, `stage2_7_multi_seed_...py` | masked strings | incremental / multi-seed generation variants |
| 3 | `stage3_analysis.py` | Stage 2 output | validity, uniqueness, similarity analysis |
| 4 | `stage4_br4_matching.py`, `stage4_1_brd4_matching.py` | generated SMILES | nearest neighbour in the BRD4 reference set |
| 5 | `stage5_chembl_matching.py`, `stage5_1chembl_matching.py` | generated SMILES | nearest neighbour among ChEMBL BRD4 actives |
| 6 | `stage6_docking.py`, `stage6_1_docking.py` | SMILES + PDB | GNINA poses, `docking_summary.csv`, score charts |
| 7 | `stage7_top_docked_report.py` | Stage 6 output | per-group PDF report |
| 8 | `stage8_novelty_potency_analysis.py`, `stage8_3_...py` | Stage 6 output | novelty–potency scatter, Pareto front, Butina clustering |

Several stages exist in more than one implementation (a bare and a `_1`/`_3`
variant). They are not always supersets of one another — read the module
docstring, which states what each one changes, before choosing.

Stage 1 is the conceptual core. It parses the PLIP XML for one binding site,
extracts that ligand from the PDB, derives a SMILES with correct bond orders,
maps PDB atom serial numbers onto SMILES atom indices, and masks either the
interacting atoms (`--mode 1`) or the non-interacting ones (`--mode 2`). Stage 1b
is the same code applied to an entire PDB mirror in parallel, with resume and
checkpointing.

---

## Property-guided fine-tuning

The generator's proposals are scored by RDKit — validity, QED, synthetic
accessibility, Tanimoto novelty against the parent, PAINS/Brenk alerts, and
optionally a learned Tox21 classifier. None of that is differentiable, so two
strategies are implemented and compared:

**Stage 9 — REINFORCE.** A policy gradient on LoRA adapters: score the sampled
completion, multiply the score by its log-probability, add a KL anchor to the
frozen pretrained model.

| script | what it is |
|---|---|
| `stage9_masked_property_finetune.py` | the reference implementation |
| `stage9_1_batched_GPU_forward_...py` | same objective, faster execution (batched forward, pooled scoring, AMP, DDP) |
| `stage9a_masked_property_without_finetuning.py` | **no training** — the untuned model measured with the identical code, the baseline every Stage 9 figure is read against |

**Stage 10 — supervised best-of-K selection.** Sample `K=16` completions, let
the chemistry score pick the best one, and train ordinary cross-entropy towards
it. The chemistry enters only through a discrete `argmin`, so nothing that
reaches `backward()` is non-differentiable.

| script | what it is |
|---|---|
| `stage10_vanila_backpropagation_training.py` | reference implementation, per-molecule, fp32 |
| `stage10_1_parallel_RDKit_scoring_resumable_training.py` | same objective, pooled RDKit scoring, step-resumable |
| `stage10_2_hardware_tuned_batched_AMP_DDP_training.py` | same objective, batched forward, AMP, DDP |
| `stage10_3_tox21_train.py` | trains the Tox21 classifier the toxicity term reads |
| `stage10_4_tox21_aware_training.py` | Stage 10.2 execution plus the learned toxicity term |
| `stage10_data_split.py` | the held-out partition every Stage 10 trainer shares |
| `stage10_lineage.py` | shared checkpoint format and run provenance |

Two properties of Stage 10 are worth stating because they shape every number it
produces:

- **Roughly 70% of molecules yield no valid completion in 16 draws** at a 15%
  mask rate, and `K=8` gives an indistinguishable rate — validity is bimodal and
  molecule-intrinsic, not a sampling-budget problem. Those molecules fall back to
  reconstructing the parent, down-weighted by `λ_fb` (default 0.3). Reconstruction
  is therefore the modal training signal, which makes memorisation the specific
  failure mode of this design.
- **The three execution paths are not bit-identical.** Batched sampling, reduced
  precision and a single dropout mask per padded forward pass each perturb the
  candidate pool, and a perturbed pool can change which candidate wins the
  `argmin` — so the model learns a different token sequence, not a numerically
  perturbed update. They are compared distributionally; exact agreement is
  asserted only in the deterministic limit (`k=1`, dropout off).

**Held-out evaluation.** `stage10_data_split.py` partitions on Bemis–Murcko
frameworks, so no validation molecule shares a ring system with a training
molecule. Group assignment is **size-blind**: groups are shuffled and dealt out
until the fold's budget is met, *taking* the group that crosses it. The earlier
largest-first rule could only hold out a group that fitted, which on this data
produced a 10% validation fold made of **four** scaffold families — an effective
sample size of 4, not 4,147. The split is computed once over every unique parent
and cached to a manifest, so it does not change with sub-sampling caps or
between variants.

---

## Baseline analysis

`baseline_analysis/` measures the **untuned** model under all three masking
strategies on 24 FDA-approved, metal-free complexes, and docks the results
against the redocked crystal ligand. It answers: are the predictions better than
the original, chemically diverse, toxic, drug-like — and does the masking
strategy change any of that?

See [`baseline_analysis/README.md`](baseline_analysis/README.md) for the full
method and [`baseline_analysis/ZIH_HPC.md`](baseline_analysis/ZIH_HPC.md) for
the TU Dresden cluster runbook.

One finding from that work applies to the whole repository: **the 15% budget is
enforced on tokens, not atoms.** Stage 9's `_remask_from_pool` draws
`floor(15% × n_tokens)` *atom indices*, but masking is token-level and one atom
can span several BPE tokens (`[C@@H]`, `[S@](=O)`), so the realised mask can
exceed the budget — measured at 15 mask tokens on an 83-token parent whose 15%
budget is 12. `baseline_analysis/baseline_common.build_mask` enforces the budget
on the realised `<mask>` count instead. The Stage 9/10 trainers still use the
atom-count rule.

---

## Self-tests

```bash
python stage1b_large_scale_PLIP_mask_calculation.py --test
python stage10_vanila_backpropagation_training.py --test
for s in build_masks generate_predictions run_docking analyze fetch_structures; do
    python baseline_analysis/$s.py --test
done
```

These need no GPU, no GNINA, no cluster and no network: docking is exercised
through a mock runner with canned poses, generation through a stub model, and
Stage 1b against a checked-in PDB/PLIP fixture.

---

## Hardware and HPC

- **Worker counts** are read from the job's actual allocation
  (`$SLURM_CPUS_PER_TASK`, then `sched_getaffinity`), never bare `os.cpu_count()`,
  which reports the whole node and over-subscribes under a smaller allocation.
- **Long runs are resumable.** Stage 1b checkpoints per PDB ID; Stages 9.1, 10.1
  and 10.2 checkpoint every N optimiser steps including Adam moments, the batch
  cursor and every RNG stream, written by atomic replace. Rebuilding Adam on each
  restart would impose a fresh adaptation transient that shows up in the training
  curve as an artefact of the interruption schedule.
- **GNINA is Linux-only.** On other platforms the docking scripts prepare every
  input and stop, so a bundle can be built on a laptop and docked on a cluster.
- **Compute nodes usually have no internet.** Download GNINA and pre-warm the
  HuggingFace cache from a login node.

---

## Repository map

```
config.py                    selector; picks config_<platform>.py and becomes it
config_colab.py              per-machine configs, 243 keys each, kept in sync
config_HPC_jupyter.py
config_laptop.py

stage0a_*.py … stage8_*.py   the generation pipeline (see Stage reference)
stage9*.py                   REINFORCE fine-tuning + the no-fine-tuning baseline
stage10*.py                  supervised best-of-K fine-tuning, split, lineage

bpe_mask_adapter.py          maps atom indices onto whole BPE token spans
hardware_autotune.py         device-aware batch/worker/precision selection
tqdm_compat.py               progress bars that behave in notebooks and logs

baseline_analysis/           untuned-model baseline: masking arms, docking, report
manuscript/                  LaTeX methods sections and references.bib
Template.tex                 team-project report
metadata.tsv                 109,597 binding sites with UniProt + FDA status
Tox21/                       Tox21 dataset and a reference model
```

---

## Known limitations

- **The central claim is not yet demonstrated.** That interaction-guided masking
  produces better molecules than random masking is the hypothesis, not a result.
  The baseline exists to test it and the docking arm has not been run at scale.
- **Vanilla ChemBERTa is a weak generator at this mask rate.** Measured on the
  24-complex baseline, ~28% of completions are RDKit-valid (PLIP++ 31.7%, PLIP−−
  25.0%, random 27.5%). Most of the pipeline's output is discarded before it is
  scored.
- **Docking scores are a proxy.** GNINA's `CNNaffinity` is a learned score, not a
  measured binding affinity, and it is not comparable across GNINA versions.
- **Metal sites are excluded, not handled.** Complexes with metal coordination
  are dropped from the baseline because GNINA scores them poorly without explicit
  parameterisation — which also drops carbonic anhydrase II, the single
  best-represented target in `metadata.tsv`.
- **`metadata.tsv` is smaller than the full PLIP corpus.** Several UniProt IDs
  quoted in project correspondence (D0VWR1, Q8DIQ1, Q8DIF8, P02945, P0A405,
  P0A407, D0VWR7) have zero rows in it, and shared IDs disagree in count. Target
  rankings computed here are rankings of *this* file.
- **Stage 4/5 are BRD4-specific.** The reference-matching stages were written
  around a BRD4 case study; the rest of the pipeline is target-agnostic.

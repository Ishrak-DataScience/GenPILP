# Baseline analysis — vanilla ChemBERTa vs. three masking strategies

Pick interesting protein–ligand complexes, mask each ligand three different ways
at **≤ 15 % of its BPE tokens**, let the **untuned** ChemBERTa fill the masks,
then dock everything and ask whether the predictions beat the original ligand.

This is the "before" picture that the Stage 9 fine-tuning is supposed to improve on.

---

## The three arms

| arm | what is masked | pool comes from |
|---|---|---|
| `plip_pos` | **PLIP ++** — the atoms PLIP reports as *contacting* the protein | complement of the PLIP−− pool |
| `plip_neg` | **PLIP −−** — the atoms PLIP reports as *not* contacting it | `config.STAGE1B_PLIP_NEGATIVE_MASK_DIR` |
| `random` | every atom, uniform | all atom indices |

Stage 1 builds its non-interaction mask as `all_atoms − interacting_atoms`, so a
PLIP−− corpus row already encodes both PLIP pools exactly. That is why no fresh
PLIP run is needed — and `build_masks.py --verify-plip-pos` proves it by running
a real Stage 1b `--mode 1` pass on the same PDB + XML and asserting the two agree.

### The budget is on tokens, not atoms

All three arms get the same budget: `floor(15 % × n_BPE_tokens)` **mask tokens**.

This deliberately differs from Stage 9's `_remask_from_pool`, which draws
`floor(15 % × n_tokens)` *atoms*. Those are not the same thing: masking is
token-level, and one atom can span several BPE tokens (`[C@@H]`, `[S@](=O)` are
2–4 tokens each). On this very selection, Stage 9's rule masked **15 tokens of an
83-token parent whose 15 % budget is 12** — about 18 % of the sequence. Since the
three arms are being *compared*, the quantity that must be held equal is how much
of the sequence the model has to reconstruct, so the budget is enforced on the
realised `<mask>` count, verified against the finished string
(`baseline_common.build_mask`, re-checked by `build_masks.assert_budget_respected`).

Effect on the real selection: mean mask size 5.79 / 5.96 / 5.96 tokens for
PLIP ++ / PLIP −− / random — comparable by construction.

---

## Running it

Everything is driven from `config.py` (`BASELINE_*` keys); every script takes
`--out-dir` to override the bundle location and `--test` to self-test offline.

```bash
# 0. one-time: where the bundle goes
export BASELINE_DIR=$(python -c "import config; print(config.BASELINE_DIR)")

# 1. choose the complexes            (laptop or cluster, ~30 s)
python baseline_analysis/select_samples.py --out-dir "$BASELINE_DIR"

# 2. freeze the masks + stage PDBs   (cluster: needs the PDB/PLIP mirrors)
python baseline_analysis/build_masks.py --out-dir "$BASELINE_DIR" --verify-plip-pos

# 3. fill the masks, vanilla model   (cluster GPU, minutes)
python baseline_analysis/generate_predictions.py --out-dir "$BASELINE_DIR"

# 4. dock parent + predictions       (cluster GPU, hours — see below)
python baseline_analysis/run_docking.py --out-dir "$BASELINE_DIR"

# 5. statistics, figures, RESULTS.md (anywhere, seconds)
python baseline_analysis/analyze.py --out-dir "$BASELINE_DIR"
```

`sbatch baseline_analysis/run_baseline_hpc.sh` runs steps 2–5 as one job.

**Cost.** The upper bound is 24 × (1 parent redock + 3 arms × 5 seeds) = 384 GNINA
runs, but the real number is far lower: vanilla ChemBERTa returns an RDKit-valid
molecule ~28 % of the time, and identical molecules are docked once per complex.
Measured on this selection: **118 docking jobs** (24 parent redocks + 94 distinct
valid predictions), roughly 1–2 GPU-hours at `--exhaustiveness 8`. Both
`run_docking.py` and `generate_predictions.py` are resumable, so a job that hits
its wall clock is simply relaunched.

**Sharded docking.** `run_docking.py --shard i --num-shards n` splits the
complexes round-robin across SLURM array tasks; each task writes its own
`docking/_shards/docking_shard_<i>.csv` and `--merge-shards` folds them into
`docking_summary.csv` afterwards (array tasks never write the shared CSV or the
manifest, which concurrent writers would corrupt). See
`run_docking_array.sh` + `run_finalize.sh`, and `ZIH_HPC.md` for the TU Dresden
walk-through.

**A note on seeded reproducibility.** Mask construction is fully deterministic
(same seeds → byte-identical `masks.csv` on any machine). Generation is seeded
per row too, but `torch.multinomial` draws from different RNG streams on CPU and
CUDA, so predictions reproduce exactly on the *same device class*, not across
one. The masks — the thing that has to be identical to compare models — are
device-independent, and their sha256 is enforced.

---

## What lands in the bundle

| file | why it exists |
|---|---|
| `selection.csv`, `candidates_all.csv`, `overview.txt` | which complexes, and the whole funnel that chose them |
| `complexes/<id>/complex.pdb`, `plip.xml`, `pools.json` | the docking input and the three atom pools, per complex |
| **`masks.csv`** | **the reproducibility artifact** — every mask, with the exact atom indices drawn |
| `mask_pools.json` | full pools, to re-derive any mask |
| `predictions.csv` | generated SMILES + QED/SA/novelty/PAINS/Brenk/(Tox21) |
| `docking_summary.csv` | every GNINA pose, plus the parent redock and the crystal-pose score |
| `per_molecule.csv`, `per_complex_delta.csv`, `arm_summary.csv`, `summary_stats.csv` | the analysis tables |
| `figures/*.png`, `RESULTS.md` | the report |
| `manifest.json` | settings, model name, and a sha256 for every artifact |

### Reproducing with a different model

The masks are frozen and hashed before any model is loaded:

```bash
python baseline_analysis/generate_predictions.py --out-dir "$BASELINE_DIR" \
       --model <other/checkpoint> --overwrite
```

`generate_predictions.py` refuses to run if `masks.csv` no longer matches the
hash recorded in `manifest.json`, so "same data, different model" is enforced
rather than assumed.

---

## How the comparison is made fair

* **Reference** = `parent_redock`: the crystal ligand's own SMILES, embedded with
  ETKDG and docked exactly like a prediction. Predictions are compared to *that*,
  not to the crystal pose, so both sides carry the same embedding error.
* **Docking-setup check**: the redocked parent's RMSD to the crystal pose (in
  place, `rdMolAlign.CalcRMS` — *not* `GetBestRMS`, which superimposes first and
  would score a pose translated out of the pocket as ~0 Å). Complexes above
  2.0 Å are flagged and every headline number is also reported without them.
* **Crystal-pose score** (`--score_only`) is recorded as a third reference, to
  show what the pocket scores when no search is involved at all.
* **Scores join by (complex, canonical SMILES)**, so a molecule proposed by two
  arms is docked once and both arms get that same score.
* **Statistics**: each complex contributes one median Δ per arm (so a complex
  with many valid predictions cannot dominate); Wilcoxon signed-rank vs 0 per
  arm; Friedman across arms; pairwise Wilcoxon with Holm correction; bootstrap
  CIs on medians. With ~24 complexes these are small-sample tests, and the report
  prints every n.

## GNINA version

**v1.3.3** (CUDA 12.8 static), fetched by `baseline_analysis/get_gnina.sh`, which
verifies the binary runs and falls back to v1.3.2's older-CUDA build if the
driver is too old. Both `config.GNINA_DOWNLOAD_URL` (Stage 6) and
`config.BASELINE_GNINA_URL` point at it as of 2026-09-17.

**Carried-over results do not survive that upgrade.** GNINA v1.3 rebuilt CNN
scoring on Torch and retrained the scoring functions on CrossDock2020 v1.3, so
CNNaffinity/CNNscore from v1.0.3 are a different quantity. Anything docked with
the old binary must be re-docked, not merged. `run_docking.py` records
`gnina --version` in the manifest so a mixed table is detectable, and warns
loudly if an SDF carries no recognised score property instead of emitting blanks.

## Self-tests

```bash
for s in build_masks generate_predictions run_docking analyze; do
    python baseline_analysis/$s.py --test || echo "FAILED: $s"
done
```

They need no cluster, no GPU, no GNINA and no network: docking is exercised
through a mock runner with canned poses, and generation through a stub model.

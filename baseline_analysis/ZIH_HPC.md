# Running the baseline on ZIH (TU Dresden) from JupyterHub

Written for whoever runs this on the ZIH cluster. It assumes a JupyterHub session
at <https://jupyterhub.hpc.tu-dresden.de> and nothing else.

Steps 1–2 (choose complexes, build masks) are **already done and shipped in the
upload bundle**, so the cluster only has to generate, dock and analyse.

> **Adjust before first run:** `baseline_analysis/zih_env.sh` — your SLURM
> account, the workspace name, and the module names `module spider Python`
> reports on your cluster. Everything else reads from that one file.

---

## 1. What to upload

Two things.

**A. The code** — if `~/GenPLIP` is already the repo, you only need the new and
changed files:

```
baseline_analysis/          (the whole folder — new)
config_HPC_jupyter.py       (changed: new BASELINE_* block)
config_colab.py             (changed: same block)
config_laptop.py            (changed: same block)
```

**B. The data bundle** — `genplip_baseline_bundle.zip` (4.3 MB), which contains:

| inside the zip | what it is |
|---|---|
| `baseline/selection.csv` | the 24 chosen complexes |
| `baseline/candidates_all.csv`, `overview.txt` | the 1,382-candidate pool and the full exclusion funnel |
| `baseline/masks.csv` | **the frozen masks** — 360 rows, all three arms, ≤15 % of tokens |
| `baseline/mask_pools.json` | the three atom pools per complex |
| `baseline/manifest.json` | settings + sha256 of masks.csv |
| `baseline/plip_neg_subset.csv` | just the 24 PLIP−− corpus rows (so the 107 MB corpus need not be copied) |
| `baseline/complexes/<id>/complex.pdb` | all 24 structures, already fetched and validated |
| `baseline/complexes/<id>/pools.json` | per-complex pools + provenance |

Nothing else is needed: no PDB mirror, no PLIP XML, no `metadata.tsv`, no
`stage1b_large_scale_plip_mask_summary.csv`.

**Upload via JupyterLab:** file browser → upload button (drag-and-drop works for
the zip). Or from your laptop:

```bash
scp genplip_baseline_bundle.zip <zih-user>@login1.barnard.hpc.tu-dresden.de:~/
scp -r baseline_analysis config_*.py <zih-user>@login1.barnard.hpc.tu-dresden.de:~/GenPLIP/
```

---

## 2. One-time setup (JupyterLab → File → New → Terminal)

```bash
# a workspace for the bundle -- NOT home, which has a small quota and this run
# writes tens of thousands of pose files
ws_allocate -F horse genplip 90
export BUNDLE="$(ws_find genplip)/baseline"

cd ~/GenPLIP
unzip -o ~/genplip_baseline_bundle.zip -d "$(dirname "$BUNDLE")"
ls "$BUNDLE"           # selection.csv masks.csv complexes/ ...

# python environment
module load release/24.04 GCC/12.3.0 Python/3.11.3      # adjust to `module spider Python`
python -m venv ~/genplip-venv
source ~/genplip-venv/bin/activate
pip install --upgrade pip
pip install rdkit torch transformers pandas numpy scipy matplotlib tqdm

# GNINA -- download on a LOGIN node; compute nodes have no direct internet.
# This fetches v1.3.3 (CUDA 12.8 static, 2.1 GB) into the workspace, verifies it
# runs, and falls back to the older-CUDA build automatically if the driver is
# too old. Takes a few minutes.
bash baseline_analysis/get_gnina.sh

# tell the job scripts about all of it
nano baseline_analysis/zih_env.sh    # set BASELINE_ACCOUNT (and check the rest)
```

**Pre-warm the model cache**, because compute nodes cannot reach HuggingFace:

```bash
source baseline_analysis/zih_env.sh && setup_baseline_env
python -c "
from transformers import AutoModelForMaskedLM, AutoTokenizer
m='seyonec/ChemBERTa-zinc-base-v1'
AutoTokenizer.from_pretrained(m); AutoModelForMaskedLM.from_pretrained(m)
print('cached into', __import__('os').environ['HF_HOME'])"
```

---

## 3. Run it

```bash
cd ~/GenPLIP
mkdir -p logs
source baseline_analysis/zih_env.sh          # exports BASELINE_DIR

# 3a. fill the 360 masks with vanilla ChemBERTa  (~1 min on GPU)
#     Note: seeded sampling reproduces exactly on the same device class; CPU and
#     CUDA draw from different RNG streams, so numbers will differ slightly from
#     the CPU rehearsal. The MASKS are identical either way (sha256-enforced).
srun -A "$BASELINE_ACCOUNT" -p "$BASELINE_PARTITION" --gres=gpu:1 \
     --cpus-per-task=4 --mem=16G --time=01:00:00 --pty \
     bash -c 'source baseline_analysis/zih_env.sh && setup_baseline_env &&
              python baseline_analysis/generate_predictions.py --out-dir "$BASELINE_DIR"'

# 3b. dock: 24 array tasks, 8 at a time
JID=$(sbatch --parsable -A "$BASELINE_ACCOUNT" -p "$BASELINE_PARTITION" \
             baseline_analysis/run_docking_array.sh)
echo "docking array: $JID"

# 3c. merge shards + build RESULTS.md, automatically after the array finishes
sbatch --dependency=afterany:$JID -A "$BASELINE_ACCOUNT" -p "$BASELINE_PARTITION" \
       baseline_analysis/run_finalize.sh
```

**How big is this really?** 24 array tasks, one per complex. The upper bound is
384 GNINA runs, but vanilla ChemBERTa's output is only ~28 % RDKit-valid and
duplicate molecules dock once, so the measured job on this bundle is **118
dockings** (24 parent redocks + 94 distinct predictions) — roughly 1–2 GPU-hours
in total, a few minutes per array task. `--array=0-23%8` keeps 8 tasks resident;
raise `%8` if your allocation allows.

Watch progress:

```bash
squeue -u $USER
tail -f logs/dock_${JID}_0.out
```

Everything is resumable. If tasks 7 and 13 hit the wall clock:

```bash
sbatch --array=7,13 -A "$BASELINE_ACCOUNT" -p "$BASELINE_PARTITION" \
       baseline_analysis/run_docking_array.sh
```

A ligand whose `docked_poses.sdf` already parses is never re-docked, so
resubmitting costs only what is genuinely missing.

---

## 4. Getting RESULTS.md

`run_finalize.sh` writes it. To produce or refresh it by hand — it is pure
CPU, seconds, and needs no GPU, so a Jupyter terminal is fine:

```bash
source baseline_analysis/zih_env.sh && setup_baseline_env
python baseline_analysis/run_docking.py --out-dir "$BASELINE_DIR" --merge-shards
python baseline_analysis/analyze.py     --out-dir "$BASELINE_DIR"
```

Output lands in `$BASELINE_DIR`:

```
RESULTS.md                     the written report (all five sections)
figures/fig1..fig5*.png        the figures
per_molecule.csv               every prediction + its best pose + its reference
per_complex_delta.csv          per (complex, arm) median Δ — what the stats use
arm_summary.csv                every headline number, machine-readable
summary_stats.csv              the same in long form
```

**Reading it in JupyterLab:** in the file browser, navigate to the workspace
(`/data/horse/ws/<user>-genplip/baseline`), then right-click `RESULTS.md` →
*Open With* → *Markdown Preview*. The figures render inline from `figures/`.
If the workspace is not visible in the browser, symlink it into home once:

```bash
ln -s "$(ws_find genplip)" ~/genplip-ws
```

**Taking it home:**

```bash
cd "$BASELINE_DIR" && tar czf ~/baseline_results.tar.gz \
    RESULTS.md figures *.csv manifest.json
# then download ~/baseline_results.tar.gz from the JupyterLab file browser
```

You can also run the analysis in a notebook cell:

```python
import os, subprocess
from IPython.display import Markdown, Image, display

# `!source ...` runs in a throwaway shell, so its exports never reach this
# kernel -- resolve the bundle path here instead.
b = subprocess.run(["ws_find", "genplip"], capture_output=True, text=True
                   ).stdout.strip() + "/baseline"
os.environ["BASELINE_DIR"] = b

subprocess.run(["python", os.path.expanduser("~/GenPLIP/baseline_analysis/analyze.py"),
                "--out-dir", b], check=True)

display(Markdown(open(f"{b}/RESULTS.md").read()))
display(Image(f"{b}/figures/fig1_docking_delta_by_arm.png"))
```

The kernel needs the same venv as the jobs: either start JupyterHub with it
selected, or run `pip install ipykernel && python -m ipykernel install --user
--name genplip` once inside `~/genplip-venv`, then pick that kernel.

---

## 5. Things that actually go wrong here

| symptom | cause and fix |
|---|---|
| `ERROR: no executable GNINA at ...` | the binary was never downloaded, or was downloaded on a compute node. Fetch it from a **login** node (step 2). |
| GNINA fails with a CUDA/driver error inside the job | the CUDA-12.8 build needs a new enough **compute-node** driver, which a login node cannot tell you. Re-fetch the older-CUDA build: `BASELINE_GNINA_FORCE_FALLBACK=1 bash baseline_analysis/get_gnina.sh`. Same v1.3 scoring functions, so results stay comparable. |
| `WARNING: docking produced poses but NO recognised score property` | a GNINA version renamed a score field. The message lists what the SDF actually contained; add that spelling to `_SCORE_ALIASES` in `run_docking.py`. Better a loud warning than a table of blanks. |
| `OSError: Can't load ... seyonec/ChemBERTa` on a compute node | the HF cache was not pre-warmed. Run the pre-warm snippet in step 2 from a login node, and keep `HF_HOME` pointed at the workspace. |
| `masks.csv has changed since build_masks.py ran` | the frozen masks were edited or re-created. That guard is deliberate — restore the uploaded `masks.csv`, or accept the new one with `--allow-mask-drift`. |
| Home quota exceeded mid-run | the bundle is in `~` instead of a workspace. Move it: docking writes an SDF per pose per ligand. |
| Analysis says "no docking results yet" | the shards were never merged. Run `--merge-shards` (step 4). |
| A complex reports `no HETATM ...` | its structure is missing the ligand copy named in the id. Re-fetch with `python baseline_analysis/fetch_structures.py --out-dir "$BASELINE_DIR"`; if it still fails, that complex is dropped and named in RESULTS.md. |

---

## 6. Which GNINA, and why it matters

| build | size | use it when |
|---|---|---|
| **v1.3.3 `gnina.cuda12.8.static`** (default) | 2.1 GB | anything current — required for H100 (Capella, sm_90), fine on A100 |
| v1.3.2 default build (automatic fallback) | 1.4 GB | the compute node's driver is too old for CUDA 12.8 |
| v1.0.3 | 306 MB | superseded — what the pipeline used before 2026-09-17 |

`get_gnina.sh` handles the choice; `config.BASELINE_GNINA_URL` /
`BASELINE_GNINA_URL_FALLBACK` hold the URLs.

**The one thing to be careful about:** GNINA v1.3 moved CNN scoring to Torch and
**retrained the scoring functions** on CrossDock2020 v1.3. A `CNNaffinity` from
v1.0.3 and one from v1.3.x are therefore different quantities. Both
`config.GNINA_DOWNLOAD_URL` (Stage 6) and `config.BASELINE_GNINA_URL` were moved
to v1.3.3 on 2026-09-17, so:

* **any Stage 6 result docked with v1.0.3 is superseded** — re-dock it rather
  than comparing it against anything produced from now on;
* `run_docking.py` records `gnina --version` in `manifest.json`, so a table built
  from two different binaries is detectable after the fact;
* do not merge shards produced by different binaries into one
  `docking_summary.csv`. If you switch versions mid-run, re-dock from scratch
  (`--overwrite`), since every comparison in the report is within one run.

This does not affect the *comparison* the baseline makes: all three arms and the
parent reference are docked by the same binary, so the Δ against the redocked
parent is internally consistent whichever version you use.

---

## 7. If you also want to rebuild the masks on the cluster

Not required — the uploaded `masks.csv` is the frozen artifact, and re-deriving
it is what the sha256 in `manifest.json` exists to detect. But if the PLIP
mirrors *are* mounted on your cluster, you can regenerate and verify them:

```bash
python baseline_analysis/build_masks.py --out-dir "$BASELINE_DIR" --verify-plip-pos
```

`--verify-plip-pos` additionally runs a real Stage 1b `--mode 1` pass per complex
and asserts PLIP++ == complement(PLIP−−). Without the mirror, point the corpus at
the shipped subset instead:

```bash
python baseline_analysis/build_masks.py --out-dir "$BASELINE_DIR" \
       --corpus "$BASELINE_DIR/plip_neg_subset.csv" --no-stage-pdb
```

Either way, re-running it invalidates the uploaded hash, so
`generate_predictions.py` will (correctly) refuse until you regenerate
predictions too.

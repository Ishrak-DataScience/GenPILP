# BRD4-GenAI: Structure-Aware Molecular Generation and Docking Pipeline

> **A fully automated, end-to-end pipeline for structure-guided de novo molecular generation targeting BRD4, combining protein–ligand interaction analysis, masked-language-model-based molecule generation, chemical similarity screening, and molecular docking.**

---

## Table of Contents

1. [Scientific Background](#1-scientific-background)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Data Requirements](#6-data-requirements)
7. [Running the Pipeline](#7-running-the-pipeline)
   - [Stage 1 — Interaction-Aware Mask Calculation](#stage-1--interaction-aware-mask-calculation)
   - [Stage 1.5 — Random Mask Generation](#stage-15--random-mask-generation)
   - [Stage 2 — Molecule Generation](#stage-2--molecule-generation)
   - [Stage 3 — Chemical Similarity Analysis](#stage-3--chemical-similarity-analysis)
   - [Stage 4 — BR4 Reference Matching](#stage-4--br4-reference-matching)
   - [Stage 5 — ChEMBL BRD4 Matching](#stage-5--chembl-brd4-matching)
   - [Stage 6 — Molecular Docking with GNINA](#stage-6--molecular-docking-with-gnina)
   - [Property-Guided Fine-Tuning — Stages 9, 9a, 9.1 and 10](#property-guided-fine-tuning--stages-9-9a-91-and-10)
8. [Smoke Tests (No Data Required)](#8-smoke-tests-no-data-required)
9. [Output Reference](#9-output-reference)
10. [Example Walkthrough](#10-example-walkthrough)
11. [Scientific References](#11-scientific-references)
12. [Acknowledgements](#12-acknowledgements)

---

## 1. Scientific Background

**BRD4 (Bromodomain-containing protein 4)** is a member of the BET (bromodomain and extra-terminal domain) family of epigenetic readers. It recognises ε-N-acetylated lysine residues on histone tails and plays a central regulatory role in transcriptional elongation, particularly of oncogenes such as *MYC*, *BCL2*, and *CCND1*. BRD4 has emerged as a high-value therapeutic target in haematological malignancies, solid tumours, inflammatory diseases, and viral infections [1,2].

JQ1, a thieno-triazolo-diazepine developed by the Bradner laboratory (2010), was the first potent, selective, and cell-permeable BRD4 inhibitor and remains the canonical reference compound for the field [3]. Since then, hundreds of BRD4 co-crystal structures have been deposited in the Protein Data Bank, making BRD4 one of the best-structurally-characterised drug targets available for computational drug design.

### Why Masked-Language-Model-Based Molecular Generation?

Traditional de novo molecular design methods (genetic algorithms, graph VAEs, reinforcement learning) treat molecular generation as a global optimisation problem and require large task-specific training runs. **ChemBERTa** [4], a transformer pre-trained on 77 million SMILES strings from the ZINC database using a masked-language-model (MLM) objective, offers a different paradigm: local, token-level perturbation of an existing ligand scaffold. By masking specific atoms in the SMILES string of a known BRD4 binder and sampling from ChemBERTa's predicted token distributions, the pipeline generates new molecules that are structurally related to the parent scaffold but chemically novel.

### Interaction-Aware vs Random Masking

The key scientific question this pipeline addresses is:

> **Does masking atoms that participate in protein–ligand interactions (interaction-aware) generate better BRD4 binders than masking randomly chosen atoms?**

To answer this, the pipeline generates two parallel sets of molecules per ligand:
- **Interaction-aware (IA):** atoms to mask are determined by PLIP [5], which identifies hydrophobic contacts, hydrogen bonds, π-stacking, salt bridges, and other non-covalent interactions from the 3D crystal structure.
- **Random:** the same number of atoms are masked uniformly at random from the full SMILES string.

Both sets are then evaluated by chemical similarity, ChEMBL activity data, and molecular docking — providing a controlled comparison of structure-guided vs. unguided generation.

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     BRD4-GenAI Pipeline                                 │
│                                                                          │
│  PDB structure   PLIP XML                                                │
│       │              │                                                   │
│       ▼              ▼                                                   │
│  ┌──────────────────────┐      ┌──────────────────────┐                 │
│  │  Stage 1             │      │  Stage 1.5            │                 │
│  │  Interaction-Aware   │      │  Random Mask          │                 │
│  │  Mask Calculation    │      │  Generation           │                 │
│  │  (PLIP-based)        │      │  (uniform random)     │                 │
│  └──────────┬───────────┘      └───────────┬──────────┘                 │
│             │ .meta.json                    │ .random.json               │
│             └──────────────┬────────────────┘                            │
│                            ▼                                             │
│                  ┌─────────────────┐                                     │
│                  │    Stage 2      │                                     │
│                  │  ChemBERTa MLM  │                                     │
│                  │  Molecule       │  incremental mask_count 1→N         │
│                  │  Generation     │  200 samples per cell               │
│                  └────────┬────────┘                                     │
│                           │  unique valid SMILES per group               │
│              ┌────────────┼─────────────────┐                           │
│              ▼            ▼                 ▼                            │
│       ┌──────────┐ ┌──────────┐    ┌──────────────┐                    │
│       │ Stage 3  │ │ Stage 4  │    │   Stage 5    │                    │
│       │ Chemical │ │  BR4 CSV │    │   ChEMBL     │                    │
│       │Similarity│ │ Nearest  │    │  BRD4 Active │                    │
│       │ Analysis │ │Neighbour │    │  Compounds   │                    │
│       └──────────┘ └──────────┘    └──────────────┘                    │
│                                                                          │
│                           ▼                                             │
│                  ┌─────────────────┐                                     │
│                  │    Stage 6      │                                     │
│                  │  GNINA Docking  │                                     │
│                  │  (autobox)      │                                     │
│                  └─────────────────┘                                     │
│                    complex PDBs + CNNscore CSV                           │
└─────────────────────────────────────────────────────────────────────────┘
```

| Stage | Script | Input | Output | Key Tool |
|---|---|---|---|---|
| 1 | `stage1_mask_calculation.py` | PDB, PLIP XML | `.meta.json` per ligand | PLIP, RDKit, SELFIES |
| 1.5 | `stage1_5_random_masking.py` | Stage-1 JSONs | `.random.json` per ligand | RDKit |
| 2 | `stage2_molecule_generation.py` | Stage-1 + 1.5 JSONs | SMILES txt files, incremental plots | ChemBERTa |
| 3 | `stage3_analysis.py` | Stage-2 SMILES txt | Molecule grids, similarity histograms | RDKit, Morgan FP |
| 4 | `stage4_br4_matching.py` | Stage-2 SMILES txt, BR4 CSV | Nearest-neighbour histograms, closeness summary | RDKit, Morgan FP |
| 5 | `stage5_chembl_matching.py` | Stage-2 SMILES txt, ChEMBL API | Same outputs as Stage 4, larger reference set | ChEMBL client, RDKit |
| 6 | `stage6_docking.py` | Stage-2 SMILES txt, PDB | Complex PDBs, docking score CSV | GNINA v1.0.3 |

---

## 3. Repository Structure

```
BRD4-GenAI/
├── config.py                      ← single file to edit before running
├── stage1_mask_calculation.py
├── stage1_5_random_masking.py
├── stage2_molecule_generation.py
├── stage3_analysis.py
├── stage4_br4_matching.py
├── stage5_chembl_matching.py
├── stage6_docking.py
│
├── Dummy_data/                    ← place your input files here
│   ├── PDB/
│   │   ├── 4QZS.pdb
│   │   ├── 3MXF.pdb
│   │   └── ...
│   ├── plip/
│   │   ├── 4QZS.xml
│   │   ├── 3MXF.xml
│   │   └── ...
│   └── BR4_PDB_Data.csv
│
└── {USER_PREFIX}/Output/          ← all outputs written here automatically
    ├── PLIP_Mask_Calculation/
    ├── Random_Mask_Calculation/
    ├── predictions_txt/
    ├── plots/
    ├── stage3_analysis/
    ├── stage4_br4_matching/
    ├── stage5_chembl_matching/
    └── stage6_docking/
```

---

## 4. Installation

### 4.1 Google Colab (Recommended)

The pipeline is designed to run in Google Colab with Google Drive as persistent storage. Open a new Colab notebook and run:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone or upload the pipeline scripts to your Drive
# Then install dependencies:
!pip install torch transformers datasets rdkit pandas seaborn matplotlib \
             selfies chembl_webresource_client -q
```

> **GPU strongly recommended for Stage 2 and Stage 6.**  
> In Colab: `Runtime → Change runtime type → T4 GPU`

### 4.2 Local Linux Installation

```bash
# Python 3.10+ required
pip install torch transformers datasets rdkit pandas seaborn matplotlib \
            selfies chembl_webresource_client

# OpenBabel (required for Stage 1 PDB → SDF conversion)
sudo apt-get install -y openbabel     # Ubuntu/Debian
# or:
conda install -c conda-forge openbabel
```

> **Note:** GNINA (Stage 6) is Linux x86_64 only. On macOS or Windows, use WSL2.

### 4.3 Dependency Reference

| Package | Version tested | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | ChemBERTa inference |
| `transformers` | ≥ 4.35 | ChemBERTa model loading |
| `rdkit` | ≥ 2023.09 | Cheminformatics, fingerprints, drawing |
| `selfies` | ≥ 2.1 | SELFIES encoding for Stage 1 masking |
| `pandas` | ≥ 2.0 | CSV handling |
| `matplotlib` | ≥ 3.7 | Plotting |
| `chembl_webresource_client` | ≥ 0.10 | Stage 5 ChEMBL API |
| `openbabel` | ≥ 3.1 | PDB → SDF conversion (Stage 1) |
| `GNINA` | v1.0.3 | Molecular docking (Stage 6, auto-downloaded) |

---

## 5. Configuration

**Edit only `config.py` — everything else updates automatically.**

```python
# config.py

BASE_DIR    = "/content/drive/MyDrive/GenAI4Drug"  # ← your Drive path
USER_PREFIX = "YourName"                            # ← your name/team
```

After editing, all output directories resolve to:
```
BASE_DIR / USER_PREFIX / Output / <stage_subdir> /
```

For example, with `BASE_DIR="/content/drive/MyDrive/GenAI4Drug"` and
`USER_PREFIX="Ishrak"`:
```
/content/drive/MyDrive/GenAI4Drug/Ishrak/Output/PLIP_Mask_Calculation/
/content/drive/MyDrive/GenAI4Drug/Ishrak/Output/predictions_txt/
...
```

### Key Tunable Parameters

| Parameter | Default | Description |
|---|---|---|
| `INCREMENTAL_NUM_SAMPLES` | `200` | ChemBERTa samples per (ligand × mask_count × strategy) cell. Raise to 500 for publication quality; lower to 20 for rapid testing. |
| `TOP_K` | `10` | Top-k sampling for ChemBERTa token prediction. |
| `TEMPERATURE` | `1.0` | Softmax temperature. Lower values (0.7) make generation more conservative. |
| `RANDOM_MASK_SEED` | `42` | Seed for reproducible random index sampling in Stage 1.5. |
| `CHEMBERTA_MODEL` | `seyonec/ChemBERTa-zinc-base-v1` | HuggingFace model identifier. |

---

## 6. Data Requirements

### 6.1 PDB Structures

Download co-crystal structures from the [RCSB Protein Data Bank](https://www.rcsb.org/). Each PDB file must contain both the receptor (ATOM records) and the bound ligand (HETATM records).

Place files as:
```
Dummy_data/PDB/4QZS.pdb
Dummy_data/PDB/3MXF.pdb
```

### 6.2 PLIP XML Files

For each PDB structure, generate a PLIP interaction report. The easiest way is the [PLIP web server](https://plip-tool.biotec.tu-dresden.de/plip-web/plip/index):

1. Upload your `.pdb` file
2. Download the XML report
3. Rename it to match the PDB ID: `4QZS.xml`
4. Place in `Dummy_data/plip/`

Alternatively, run PLIP locally:
```bash
pip install plip
plip -f 4QZS.pdb -x -o ./plip_output/
```

### 6.3 BR4 Reference CSV (`BR4_PDB_Data.csv`)

A CSV of known BRD4 ligands with columns:

```
PDB ID, Ligand Code, Ligand Chain, UniProt ID, Smiles, Lig_ChEMBL_ID, Chembl ID
5A5S, HOH, A, O60885, O, CHEMBL1098659, CHEMBL1163125
5A5S, EDO, A, O60885, OCCO, CHEMBL457299, CHEMBL1163125
```

Place as `Dummy_data/BR4_PDB_Data.csv`.

### 6.4 Registering Ligands in `config.py`

Add each ligand to `PIPELINE_INPUTS` in `config.py`:

```python
PIPELINE_INPUTS = [
    {
        "pdb_path":      BASE_PDB_PATH + "4QZS",
        "plip_xml_path": BASE_XML_PATH + "4QZS",
        "resname": "JQ1",   # 3-letter residue name from the PDB HETATM record
        "chain":   "A",     # chain identifier
        "resseq":  201,     # residue sequence number
    },
    # add more entries here ...
]
```

> **Tip:** To find `resname`, `chain`, and `resseq` for your ligand, either open the PDB file and search HETATM records, or upload to the [PLIP web server](https://plip-tool.biotec.tu-dresden.de/plip-web/plip/index) and read the binding-site panel on the left.

---

## 7. Running the Pipeline

Run the stages in order. Each stage has an interactive smoke test — type `yes` at the first prompt to verify the installation without needing real data.

---

### Stage 1 — Interaction-Aware Mask Calculation

**What it does:** Parses PLIP XML files to identify which ligand atoms form non-covalent interactions with the receptor (hydrophobic contacts, H-bonds, π-stacking, salt bridges, etc.). Those atom indices are stored in `.meta.json` files, which are the input to all subsequent stages.

```bash
python stage1_mask_calculation.py
```

**Output:**
```
Output/PLIP_Mask_Calculation/
  JQ1_A_201_masked.selfies_JQ1_A_201.meta.json
  JQ1_A_201.2d_interactions.png          ← 2D interaction diagram
  ...
```

Each `.meta.json` contains:
```json
{
  "smiles": "CC1=NN=C2N1CC(=CC2=O)c3ccc(Cl)cc3",
  "masked_atom_indices": [0, 3, 7, 11],
  "masking_mode": "attractive",
  "ligand": {"resname": "JQ1", "chain": "A", "resseq": 201}
}
```

---

### Stage 1.5 — Random Mask Generation

**What it does:** For each Stage-1 JSON, samples the same number of atom indices uniformly at random (seeded for reproducibility). Creates the control arm against which interaction-aware generation is compared.

```bash
python stage1_5_random_masking.py
```

**Output:**
```
Output/Random_Mask_Calculation/
  JQ1_A_201_masked.selfies_JQ1_A_201.meta.random.json
```

---

### Stage 2 — Molecule Generation

**What it does:** For each ligand, iterates `mask_count = 1, 2, … N` where N is the number of interaction-aware atom indices. At each step, generates molecules by:
1. Masking the first `mask_count` atoms (IA or random)
2. Running ChemBERTa sequential infilling (200 samples per cell)
3. Validating candidates with RDKit
4. Plotting unique valid SMILES count vs mask count for both strategies

```bash
python stage2_molecule_generation.py
```

At launch you will be prompted:

```
  Run the smoke test? [y/N]: n

  Before running the full pipeline ...
  Run the smoke test? [y/N]: n

  Number of Masks on x-axis, Unique Valid SMILES on y-axis
  [options explained]
  Generate per-mask-count outputs? [Y/n]: Y
  Deduplicate pooled IA + random? [Y/n]: Y
```

**Output:**
```
Output/predictions_txt/
  JQ1-A-201/
    ia_mask001.txt      ← unique valid SMILES for IA, mask_count=1
    rand_mask001.txt    ← unique valid SMILES for random, mask_count=1
    ia_mask002.txt
    ...
Output/plots/
  JQ1-A-201_incremental_masking.png
```

**Example plot interpretation:**  
If the IA line rises faster than the random line as mask count increases, interaction-aware masking generates more diverse valid molecules — suggesting the model is guided by chemically meaningful positions.

---

### Stage 3 — Chemical Similarity Analysis

**What it does:** For each ligand group, pools generated molecules and computes:
- **Pairwise Tanimoto histogram:** how similar are generated molecules to each other? (diversity metric)
- **vs-original Tanimoto histogram:** how similar are generated molecules to the parent ligand? (scaffold retention metric)
- **Molecule grid:** structural visualisation of all generated molecules

Fingerprint: Morgan (circular), radius = 2, 2048 bits (consistent with standard QSAR practice [6]).

```bash
python stage3_analysis.py
```

**Output:**
```
Output/stage3_analysis/
  JQ1_A_201/
    aggregated/
      molecule_grid.png
      histogram_pairwise.png
      histogram_vs_original.png
    mask_001/         ← per-step outputs (if requested)
      ...
```

---

### Stage 4 — BR4 Reference Matching

**What it does:** For each generated molecule, finds its single nearest neighbour in `BR4_PDB_Data.csv` by Tanimoto similarity and answers:

> *"What fraction of generated molecules from group EAM-A-1 are most similar (≥ T) to the same BR4 reference ligand?"*

```bash
python stage4_br4_matching.py
```

Prompts:
```
  Pool choice (ia / rand / both) [both]:
  Minimum heavy atoms for BR4 ligands [7]:
  Similarity threshold T (0.0–1.0) [0.4]:
```

**Example `closeness_summary.txt`:**
```
  Source ligand  : EAM-A-1
  Generated mols : 12
  Threshold T    : 0.40

  ✓ 9/12 (75%) generated molecules are most similar to BR4 ligand 'JQ1'
    (Tanimoto ≥ 0.40, median similarity = 0.61).
    3/12 have no BR4 match above threshold.
```

---

### Stage 5 — ChEMBL BRD4 Matching

**What it does:** Identical analysis to Stage 4 but uses the full ChEMBL BRD4 active compound set (target `CHEMBL1163125`) as the reference — potentially thousands of molecules vs. the smaller BR4 CSV. Molecules are fetched once and cached to Drive for fast re-runs.

```bash
pip install chembl_webresource_client  # if not already installed
python stage5_chembl_matching.py
```

Additional prompt:
```
  pChEMBL threshold [5.0]:    ← 5.0 = IC50 ≤ 10 µM
```

On first run, ChEMBL is queried and the result is cached at:
```
Output/stage5_chembl_matching/chembl_brd4_cache.csv
```

Subsequent runs reuse the cache (prompt: `Use cached ChEMBL data? [Y/n]`).

---

### Stage 6 — Molecular Docking with GNINA

**What it does:** Docks all generated molecules into the BRD4 binding pocket using GNINA [7] with the autobox feature — the binding box is automatically defined from the original co-crystallised ligand. Outputs complex PDB files and a unified docking score CSV.

```bash
python stage6_docking.py
```

> **GNINA is auto-downloaded (~500 MB) to `BASE_DIR/gnina` on first run.**  
> In Colab, it is then copied to `/content/gnina` for execution (required because Google Drive is FUSE-mounted and cannot execute binaries directly).

Prompts:
```
  Run the smoke test? [y/N]: n
  Number of poses to keep (1–9) [1]:
  Pool choice (ia / rand / both) [both]:
```

**Output:**
```
Output/stage6_docking/
  docking_summary.csv           ← all groups, molecules, poses, scores
  JQ1_A_201/
    rec.pdb                     ← receptor (ATOM records only)
    orig.pdb                    ← autobox ligand
    mol_0001/
      ligand.sdf                ← 3D input conformer (ETKDG + UFF)
      docked_poses.sdf          ← GNINA multi-pose output
      complex_pose001.pdb       ← receptor + docked ligand
      docking.log
```

**CSV columns:**

| Column | Description |
|---|---|
| `ligand_group` | Source ligand (e.g. `JQ1-A-201`) |
| `mol_idx` | Index within the group |
| `smiles` | Generated SMILES |
| `pose` | Pose number (1 = best) |
| `CNNscore` | CNN binding probability [0,1] — higher is better |
| `CNNaffinity` | CNN-predicted −log Kd/Ki — higher is better |
| `minimizedAffinity` | Vinardo score (kcal/mol) — more negative is better |
| `complex_pdb` | Path to complex PDB file |

### Property-Guided Fine-Tuning — Stages 9, 9a, 9.1 and 10

Stages 1–6 generate and dock molecules with a **frozen** ChemBERTa. Stages 9 and 10 instead **fine-tune** ChemBERTa so that the molecules it generates score better on chemical properties — validity, drug-likeness (QED), synthetic accessibility (SA), novelty relative to the parent, and structural toxicity alerts (PAINS/Brenk).

They are two different answers to one hard problem, and the comparison between them is the point.

#### The problem both stages solve

Every property above is computed by running RDKit on a **decoded SMILES string**. Turning token logits into a string (sample → decode → `Chem.MolFromSmiles`) is not differentiable, so a number RDKit returns has no gradient. You cannot simply write `loss = 1000 if invalid` and call `.backward()` — that is a Python float, not a tensor.

| | how the RDKit number reaches the weights |
|---|---|
| **Stage 9** | **REINFORCE.** Multiply the score by `Σ log P(chosen token)`, which *is* differentiable. The score enters as a scalar multiplier and is never differentiated. |
| **Stage 10** | **Selection.** Sample K completions, score each with RDKit, and train ordinary **cross-entropy** toward the tokens of the best one. RDKit picks *which* tokens to pull toward; the gradient is plain supervised CE. |

#### The scripts

| script | trains? | method | objective |
|---|---|---|---|
| `stage9a_masked_property_without_finetuning.py` | **no** | — | baseline: measures the untrained model |
| `stage9_masked_property_finetune.py` | yes | REINFORCE | **maximize** a score in [0, 1] |
| `stage9_1_batched_..._Lora_finetuning.py` | yes | REINFORCE | identical objective to Stage 9, ~58× faster |
| `stage10_vanila_backpropagation_training.py --variant a` | yes | supervised best-of-K | **minimize** a loss |
| `stage10_vanila_backpropagation_training.py --variant b` | yes | supervised best-of-K | **minimize** a loss |
| `stage10_1_parallel_RDKit_scoring_resumable_training.py` | yes | supervised best-of-K | identical objective **and identical bits** to Stage 10 |
| `stage10_2_hardware_tuned_batched_AMP_DDP_training.py` | yes | supervised best-of-K | identical objective to Stage 10, distributionally |

**Stage 9.1 is not a different experiment.** It trains the same objective on the same data with the same estimator; only the execution differs (batched GPU forward, parallel RDKit scoring, optional multi-GPU via `torchrun`). Its adapter is directly comparable to Stage 9's. Use it when Stage 9 is too slow.

Every one of those speedups is opt-in and individually switchable — see [Hardware](#hardware--you-choose-which-speedups-stage-91-is-allowed). `config.STAGE9_1_SPEED = "off"` (or `--speed off`) runs Stage 9's exact execution path from this script, which is the setting to use when the run exists to be compared rather than to finish quickly.

#### Stage 9 vs Stage 10 — the differences that matter

| | **Stage 9 / 9.1** | **Stage 10a / 10b** |
|---|---|---|
| training signal | REINFORCE (policy gradient) | cross-entropy (supervised) |
| direction | maximize score, higher = better | minimize loss, lower = better |
| what `.backward()` sees | `−(score − baseline) · Σ log P` | `F.nll_loss(log_probs, best_candidate)` |
| trainable weights | LoRA rank-8 on query/value in **all** blocks — **147k params (0.33%)** | LM head + last encoder block — **7.7M params (17.4%)** |
| anchor to pretrained | explicit KL penalty (`STAGE9_KL_BETA`) | implicit — CE targets are always real molecules |
| unbounded failure mode | **yes** — persistent negative advantage drives `log P → −∞` | **no** — CE is bounded below by 0 |
| terms scored | 6 defined, **5 live** (`W_TOX21 = 0`) | 5 (Tox21 not implemented) |
| forward passes / molecule | 2 (policy + KL reference) | 1 (serves both sampling and CE) |

Note the capacity difference: Stage 9 trains 0.33% of the model, Stage 10 trains 17.4%. A difference in results reflects **both** the objective and the capacity budget, not the objective alone. State that when reporting.

#### Stage 10a vs 10b

Both pick the lowest-loss candidate and train CE toward it. They differ in one term:

- **10a** — CE **plus an unlikelihood term** `−log(1 − P(token))` that actively pushes *down* the tokens of invalid candidates. This is "learn not to generate this SMILES".
- **10b** — CE only. Bad candidates simply never become targets.

Tokens that equal the target at the same position are excluded from the push-down, so the two terms never fight.

**Why this comparison is interesting.** Measured on this data at 15% masking, **~70% of molecules produce zero valid completions out of 16** (K = 8 gives the same rate — it is molecule-intrinsic, not sample luck). On those molecules 10b has only the down-weighted parent target, so it becomes mostly reconstruction; 10a still gets real signal from pushing down 16 known-invalid completions. That is likely the dominant effect you will measure.

#### The Stage 10 loss

```
valid:    w_qed*(1 - QED) + w_sa*(SA_raw - 1)/9
        + w_novelty*similarity(parent, generated) + w_tox_alert*alert
invalid:  S + w_valid,   where S = w_qed + w_sa + w_novelty + w_tox_alert
```

With the shipped weights `S = 1.0`, so a valid molecule scores in `[0, 1]` and an invalid one scores `2.0` — **strictly worse than every valid molecule**, by a margin of exactly `w_valid`. That encodes "validity matters most" without a 1000× term that would swamp the cross-entropy (which is O(1–10)) and destabilize the step.

#### The parent fallback

When no candidate is valid, Stage 10 falls back to the **parent's own tokens** as the target — always available and valid by construction. But reconstructing the parent scores `similarity = 1`, the *worst* possible novelty. At full weight, ~70% of steps would train a copier.

`STAGE10_FALLBACK_WEIGHT` (default `0.3`) scales those steps only, and spans the whole design space:

| value | behaviour |
|---|---|
| `1.0` | train them fully — fastest validity gain, high risk of collapsing to copying |
| `0.3` | **default** — reconstruction acts as a background regularizer |
| `0.0` | equivalent to skipping those molecules entirely |

Watch the `novelty` term in the per-epoch loss split. If it climbs toward its full weight (0.30) while validity improves, the model is becoming a copier — lower this knob.

#### One objective, three execution paths

The supervised best-of-K objective is implemented once and executed three ways. **Same loss, same data, same K, same optimizer-step count** — they differ only in how the work is arranged, and in how much of a crash they can survive.

| script | speciality | reproduces Stage 10 | resumes at |
|---|---|---|---|
| `stage10_vanila_backpropagation_training.py` | **The reference implementation.** One forward per molecule, one RDKit call per candidate, fp32, single process. Simplest thing that is obviously correct. | — (it *is* the reference) | epoch boundaries |
| `stage10_1_parallel_RDKit_scoring_resumable_training.py` | **Step-resumable, with RDKit scoring in a process pool.** Adds the one speedup that cannot move a number, plus Adam's moments and the RNG streams in every checkpoint. | **bit-identical** | any optimizer step |
| `stage10_2_hardware_tuned_batched_AMP_DDP_training.py` | **Batched forward, mixed precision, length bucketing and multi-GPU.** Everything Stage 9.1 does, applied to this objective. | distributionally | any optimizer step |

**Why the middle one stops where it does.** RDKit runs on decoded strings, so moving it to another process is a pure function evaluated elsewhere — it cannot change a float. The pooled loss equals the serial loss to *zero* tolerance, which its self-test asserts. That is worth having on its own: at K = 16 and batch 16 there are **256 RDKit measurements per optimizer step**, ~2.3 s of pure RDKit, which is where nearly all the wall-clock goes.

**Why the third one cannot make that promise, and why that is specific to this objective.** In Stage 9 the score *multiplies* a log-probability, so a differently-drawn sample is still an unbiased estimate. Here the score **selects the training target**. Three things break the correspondence:

- **Batched sampling** — one `multinomial` over every mask in the batch consumes the RNG in a different order than one call per molecule.
- **Mixed precision** — bf16/fp16 moves the logits in the last bits.
- **Dropout** — draws once per *forward*, so B separate forwards consume B masks and one padded forward consumes one. Measured here, this alone shifts the mask logits by ~1.2, **larger than anything AMP does**, and no seed can align it.

Any of those can make a different candidate win best-of-K, at which point the model trains toward a **different token sequence** — not a rounded gradient, a different target. So compare the tuned path against the reference *distributionally* (mean validity, mean loss, mean novelty over many molecules), never by diffing outputs.

That trade is reversible per run:

```bash
python stage10_2_hardware_tuned_batched_AMP_DDP_training.py --speed off    # bit-identical to Stage 10
python stage10_2_hardware_tuned_batched_AMP_DDP_training.py --speed safe   # + the pool only; still bit-identical
python stage10_2_hardware_tuned_batched_AMP_DDP_training.py                # "fast" (default)
```

#### One output folder for the family, or three

By default each of the three writes to its own directory and resumes only its own checkpoints — three independent runs of one objective, which is what makes a timing comparison between them a comparison of execution and nothing else.

```python
# config.py
STAGE10_SHARED_OUTPUT = False                          # the default
STAGE10_SHARED_DIR    = f"{_OUT}/stage10_family_shared/"
```

Set it to `True` (or pass `--shared` / `--separate` to any of the three) and all three write to `STAGE10_SHARED_DIR/variant_<v>/` and share **one continuable checkpoint**, `stage10_lineage.pt`. A run interrupted under any of them resumes under any other, at the exact batch it stopped on, with Adam's moments and the RNG streams intact — start on the reference implementation, move to the tuned one when a GPU frees up, finish wherever.

```
stage10_family_shared/variant_a/
├── stage10_lineage.pt          weights + Adam + history + cursor + RNG + provenance
├── stage10_lineage.json        the same minus tensors — readable with `cat`
├── epoch_001.pt …              per-epoch snapshots
├── config.json, model.safetensors        ONE lineage model (the endpoint)
├── stage10a_training_curves.png          per-stage figures, never overwritten
├── stage10a_tox_alert_rate.png           PAINS/Brenk alert rate vs epoch
├── stage10.1a_training_curves.png
├── stage10.1a_tox_alert_rate.png
├── stage10.2a_training_curves.png
└── stage10.2a_tox_alert_rate.png
```

Three consequences worth knowing before you turn it on:

- **The reference implementation gains optimizer state.** Its own format saves the unfrozen tensors and nothing else, so *every* resume today silently rebuilds Adam from scratch and discards `exp_avg` / `exp_avg_sq`. In shared mode it writes and reads the lineage checkpoint instead, and that stops. With the flag off, its existing `epoch_NNN.pt` + JSON path is untouched.
- **One model at the root**, whichever stage finished the lineage last — the correct reading if the three are continuing a single run. Training curves and property figures keep their per-stage filenames, so no diagnostic is lost.
- **A lineage may mix execution paths, and that changes what the run is.** Epochs trained under the tuned path's batched bf16 sampling are not epochs trained under the reference path's fp32 per-molecule sampling. Every save therefore appends to a **provenance log**, and a resume across a change prints:

```
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  MIXED-EXECUTION LINEAGE
    checkpoint written by : stage10, fp32, per-molecule sampling
    this run executes as  : stage10_2, speed=fast, bf16, batched sampling
    changed               : stage, amp, batched
    ...
    epoch 1          steps 0-812      stage10, fp32, per-molecule sampling
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

The log lives in plain text in `stage10_lineage.json` next to the weights, because `model.safetensors` has no memory of the dtype it was trained under and the terminal scrollback will be gone by the time anyone asks. **Variants never merge** in either mode: 10a and 10b are different objectives, so each keeps its own `variant_<v>` subdirectory. Only execution is allowed to vary along a lineage.

#### Running a controlled comparison

Every training run below shares the same data, the same masking rate, and the same evaluation figures, so only the training strategy differs. Within supervised best-of-K, the three execution paths share the objective too — so a difference between *those* is wall-clock, not learning.

```bash
# 1. Baseline first — the untrained model, and the reference every other
#    figure is compared against.
python stage9a_masked_property_without_finetuning.py

# 2. REINFORCE. Use 9.1 unless you specifically want the original loop.
python stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py
#    or:  python stage9_masked_property_finetune.py
#    9.1's speedups are opt-in: add --speed off for Stage 9's exact execution
#    path, --speed safe for only the results-preserving ones. See Hardware.

# 3. Supervised best-of-K, both variants (separate output directories).
#    The reference implementation — simplest, slowest, resumes per epoch:
python stage10_vanila_backpropagation_training.py --variant a
python stage10_vanila_backpropagation_training.py --variant b

#    Same bits, but scored across a process pool and resumable per step.
#    Prefer this over the reference unless you need the reference itself:
python stage10_1_parallel_RDKit_scoring_resumable_training.py --variant a

#    Same objective with every hardware speedup — use it when the run has to
#    finish, and compare it distributionally rather than by diffing:
python stage10_2_hardware_tuned_batched_AMP_DDP_training.py --variant a
torchrun --nproc_per_node=4 stage10_2_hardware_tuned_batched_AMP_DDP_training.py
```

Every script accepts `--limit N` (pairs to score per source in the final property pass; `--limit none` for uncapped) and `--seed N`.

> **Comparability rule.** `config.STAGE9N9A_EVAL_MAX_PAIRS_PER_SOURCE` is part of what a property figure *means*. Every run you intend to compare must use the **same** value — the sample is seeded, so a capped run scores a deterministic *subset* of an uncapped one, and the two are not comparable. The figure's own footer records which mode produced it (`"randomly sampled N of M"` appears only on a capped figure).

Likewise `config.MASK_PERCENT` (default 15) is a **single shared value** read by Stages 9, 9a and 10. Changing it changes what all of them mean; change it once and re-run everything you intend to compare.

#### Key configuration

```python
# config.py
MASK_PERCENT = 15                     # shared by Stage 9, 9a and 10

# Stage 9 — score weights (maximized, sum to 1.0)
STAGE9_SCORE_W_VALID     = 0.25
STAGE9_SCORE_W_QED       = 0.20
STAGE9_SCORE_W_SA        = 0.15
STAGE9_SCORE_W_NOVELTY   = 0.15
STAGE9_SCORE_W_TOX_ALERT = 0.25
STAGE9_SCORE_W_TOX21     = 0.00       # 0 until a Tox21 checkpoint is trained
STAGE9_KL_BETA           = 0.05       # anchor to the pretrained distribution
STAGE9_MAX_PAIRS_PER_PARENT = 3       # see note below

# Stage 9.1 — which speedups you allow (see Hardware, below)
STAGE9_1_SPEED = "fast"               # "off" | "safe" | "fast" | "auto"

# Stage 10 — loss weights (minimized; the first four define S)
STAGE10_VARIANT             = "a"     # "a" = CE + unlikelihood, "b" = CE only
STAGE10_NUM_CANDIDATES      = 16      # K
STAGE10_W_QED               = 0.30
STAGE10_W_SA                = 0.20
STAGE10_W_NOVELTY           = 0.30
STAGE10_W_TOX_ALERT         = 0.20
STAGE10_W_VALID             = 1.00    # margin by which invalid beats worst valid
STAGE10_FALLBACK_WEIGHT     = 0.3
STAGE10_UNLIKELIHOOD_WEIGHT = 0.5     # variant "a" only
STAGE10_UNFREEZE_LAST_N_BLOCKS = 1    # + LM head, always

# Stage 10 family — one output folder and one continuable checkpoint, or three
STAGE10_SHARED_OUTPUT = False         # True = shared lineage; see above

# Supervised best-of-K, hardware-tuned execution — which speedups you allow
STAGE10_2_SPEED = "fast"              # "off" | "safe" | "fast" | "auto"
```

> **`STAGE9_MAX_PAIRS_PER_PARENT` is the most important knob in this block.** Stage 1b writes one row per ligand *instance*, and PDB instance counts follow crystallography, not chemistry. Uncapped, sulfate, glycerol, NAG and acetate alone are **33% of the training set** and the top 100 parents are **58.5%** — the gradient would be dominated by buffer components and cryoprotectants. The cap keeps at most N instances per unique canonical parent (335k pairs → 65k at N=3).

#### What to expect

Measured on this dataset with the untrained model, one-shot decoding, 15% masking, unique parents:

| | value |
|---|---|
| baseline validity | ~22% |
| molecules with no valid candidate out of 16 | ~70% |
| effect of temperature / top-k on validity | **none** (greedy gives 28.4%, `T=1.2 k=20` gives 29.6%) |
| effect of mask percentage | **large** — 5%: 41%, 10%: 27%, 15%: 22%, 25%: 16% |

Validity is the binding constraint on this data, and **mask percentage is the only lever that moves it substantially**. If results disappoint, lower `MASK_PERCENT` before tuning anything else — but re-run every stage afterwards.

#### Hardware — you choose which speedups Stage 9.1 is allowed

`hardware_autotune.py` detects CPU allocation (SLURM / cgroup / affinity — not `os.cpu_count()`, which reports the host, not your job), RAM limits and GPU compute capability, and derives precision, worker counts and batch size from them:

```bash
python hardware_autotune.py     # the machine, and what it would pick
```

**Stage 9.1 never lets that detection decide anything on its own.** Every execution choice the stage makes is a config knob, and the hardware profile is consulted *only* where you wrote `"auto"`. One preset sets them all; any individual knob overrides it:

```
--speed / --workers / --no-batch   >   config.STAGE9_1_<KNOB>   >   config.STAGE9_1_SPEED
```

| `STAGE9_1_SPEED` | 7a batched rollout | 7b RDKit pool | AMP | TF32 | length bucketing | CPU threads |
|---|---|---|---|---|---|---|
| `"off"` | off | serial | fp32 | off | off | untouched |
| `"safe"` | off | auto | fp32 | off | off | auto |
| `"fast"` **(default)** | auto | auto | auto | on | on | auto |
| `"auto"` | alias of `"fast"` | | | | | |

- **`"off"` is the parity setting** — Stage 9's per-molecule rollout, serial scoring, full fp32, TF32 explicitly disabled. Expect it to be several times slower; that is the price of the comparison.
- **`"safe"`** allows only speedups that cannot change *which* molecules get sampled or what the objective is: the RDKit process pool (pinned by self-test `[4]` — pooled `==` serial to 1e-9), plus thread and pool tuning. *Honest caveat:* a different CPU thread count can reorder BLAS reductions, so a **CPU-only** run may differ in the last bits. On GPU nothing changes.
- **`"fast"`** is exactly what this stage did before the knobs existed. It is *distributionally* identical to `"off"` — same objective, same estimator — but **not token-for-token identical**, because batching consumes the RNG stream in a different order. That is expected, not a defect, and the script's docstring explains why.

Set any knob individually and leave the rest at `None`:

```python
# config.py
STAGE9_1_SPEED            = "fast"   # the preset; every None below follows it

STAGE9_1_BATCHED_ROLLOUT  = None     # 7a   "auto" | True | False
STAGE9_1_SCORING_WORKERS  = None     # 7b   "auto" | N | 0 for serial
STAGE9_1_AMP              = None     # "auto" | "bf16" | "fp16" | False
STAGE9_1_TF32             = None     # True | False
STAGE9_1_LENGTH_BUCKETING = None     # True | False
STAGE9_1_TORCH_THREADS    = None     # "auto" | N | 0 (leave torch alone)
STAGE9_1_TOX21_SUBBATCH   = None     # rows per Tox21 forward; 0 = whole batch
STAGE9_1_MAX_SEQ_TOKENS   = None     # rollout truncation ceiling
STAGE9_1_DDP_SPLIT_BATCH  = None     # batch size is GLOBAL (True) or per rank
# also: _GRAD_SCALER (+_INIT_SCALE), _MATMUL_PRECISION, _WORKER_BLAS_THREADS,
#       _POOL_START_METHOD, _POOL_CHUNK_FACTOR, _POOL_MAXTASKSPERCHILD,
#       _DDP_FIND_UNUSED, _DDP_BACKEND
```

Before trusting `"auto"` on a new machine, print what it resolves to *there*:

```bash
# machine profile AND every knob's resolved value, then exit — no training
python stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py --hardware

# time one preset against another without editing config between runs
python stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py --speed off
```

`config.py` alone cannot tell you what a run will do, because of the `"auto"`s — `--hardware` can. Every training run also prints the same resolved values in its banner, so a saved log answers "which speedups did this run actually use?" on its own.

> **`STAGE9_1_BATCH_SIZE` is deliberately *not* part of the preset.** Batch size here is a **learning** knob, not a speed one: it sets the optimizer-step count, and REINFORCE is a high-variance estimator that needs steps. A preset that quietly quartered the learning in exchange for wall-clock would be lying about what it does. If you raise it, raise `STAGE9_LEARNING_RATE` roughly in proportion or raise `STAGE9_NUM_EPOCHS`, and watch the reward curve.

> **Two behaviours changed when these knobs were added.** `STAGE9_1_TF32 = False` previously could not turn TF32 off — the hardware profile switched it on *before* the flag was read, so the banner printed `OFF` over kernels that were still using it. And pool workers never pinned their BLAS threads, so W workers each started a full-size thread pool on a W-core allocation. Both are fixed. Stock defaults are otherwise unchanged, so existing runs are unaffected.

#### Hardware — the same choice, for supervised best-of-K

The reference best-of-K implementation is deliberately single-process and untuned. The step-resumable one takes exactly one speedup — the RDKit pool — and keeps its worker count in `STAGE10_1_SCORING_WORKERS`; it borrows Stage 9.1's pool *machinery* (and therefore its `STAGE9_1_POOL_*` and `STAGE9_1_WORKER_BLAS_THREADS` tuning) but is not governed by `STAGE9_1_SPEED`.

The hardware-tuned one has the full knob set, in its own `STAGE10_2_*` namespace with the same three-level precedence. It never inherits Stage 9.1's preset — one stage silently adopting another stage's speed setting is precisely the confusion these knobs exist to remove.

```
--speed / --workers / --no-batch   >   config.STAGE10_2_<KNOB>   >   config.STAGE10_2_SPEED
```

| `STAGE10_2_SPEED` | batched forward | RDKit pool | AMP | TF32 | length bucketing | CPU threads | = Stage 10? |
|---|---|---|---|---|---|---|---|
| `"off"` | off | serial | fp32 | off | off | untouched | **bit-identical** |
| `"safe"` | off | auto | fp32 | off | off | auto | **bit-identical** |
| `"fast"` **(default)** | auto | auto | auto | on | on | auto | distributionally |
| `"auto"` | alias of `"fast"` | | | | | | |

- **`"off"` is the parity setting.** Its self-test asserts the loss equals the reference implementation's to *zero* tolerance, serial and pooled alike.
- **`"safe"` keeps that guarantee** while taking the speedup that actually matters here — RDKit is ~16× more dominant in this objective than in Stage 9, because K = 16 candidates are scored per molecule instead of one.
- **`"fast"`** batches the forward *and the sampler*, which is what forfeits bit-identity. See [One objective, three execution paths](#one-objective-three-execution-paths) for why that costs more here than it does in Stage 9.1.

```python
# config.py — set any knob individually, leave the rest at None
STAGE10_2_SPEED            = "fast"   # the preset; every None below follows it

STAGE10_2_BATCHED_FORWARD  = None     # "auto" | True | False  ← breaks bit-identity
STAGE10_2_SCORING_WORKERS  = None     # "auto" | N | 0 for serial
STAGE10_2_AMP              = None     # "auto" | "bf16" | "fp16" | False
STAGE10_2_TF32             = None     # True | False
STAGE10_2_LENGTH_BUCKETING = None     # True | False
STAGE10_2_TORCH_THREADS    = None     # "auto" | N | 0 (leave torch alone)
STAGE10_2_MAX_SEQ_TOKENS   = None     # padded-forward truncation ceiling
STAGE10_2_DDP_SPLIT_BATCH  = None     # batch size is GLOBAL (True) or per rank
STAGE10_2_BATCH_SIZE       = None     # None inherits STAGE10_BATCH_SIZE;
                                      # int | "auto" (token budget) | "hardware"
# also: _GRAD_SCALER (+_INIT_SCALE), _MATMUL_PRECISION, _WORKER_BLAS_THREADS,
#       _POOL_START_METHOD, _POOL_CHUNK_FACTOR, _POOL_MAXTASKSPERCHILD,
#       _DDP_FIND_UNUSED, _DDP_BACKEND, _MAX_BATCH_TOKENS, _MAX_BATCH_MOLECULES,
#       _HW_MIN_TOTAL_STEPS, _HW_MAX_BATCH, _HW_MEM_HEADROOM,
#       _CHECKPOINT_EVERY_STEPS, _KEEP_EPOCH_CHECKPOINTS, _SEED
```

```bash
# machine profile AND every knob's resolved value, then exit — no training
python stage10_2_hardware_tuned_batched_AMP_DDP_training.py --hardware
```

That report includes a `stage10_identical` line — a single yes/no answer to "is this run still comparable to the reference bit-for-bit?", which no reading of `config.py` alone can give you.

> **`STAGE10_2_BATCH_SIZE` is deliberately *not* part of the preset**, for the same reason as Stage 9.1's: it sets the optimizer-step count, so a preset named `"fast"` that quietly quartered the number of updates would be measuring something other than speed. Under `torchrun` it is treated as the **global** batch and split across ranks, so the step count matches a single-GPU run and the multi-GPU gain is real wall-clock rather than fewer, larger updates.

---

## 8. Smoke Tests (No Data Required)

Every stage ships with a self-contained smoke test that creates synthetic inputs, runs the full code path, and verifies all outputs. No PDB files, no PLIP XML, no real GNINA binary, no network connection required.

```bash
python stage1_5_random_masking.py    # has built-in unit test — see docstring
python stage2_molecule_generation.py # → answer "yes" at the prompt
python stage3_analysis.py            # → answer "yes" at the prompt
python stage4_br4_matching.py        # → answer "yes" at the prompt
python stage5_chembl_matching.py     # → answer "yes" (uses synthetic CSV, no ChEMBL call)
python stage6_docking.py             # → answer "yes" (uses mock GNINA shell script)

# Fine-tuning stages — fully self-contained, no prompt, no data needed:
python stage9_masked_property_finetune.py --test
python stage9a_masked_property_without_finetuning.py --test
python stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py --test
python stage10_vanila_backpropagation_training.py --test   # covers BOTH 10a and 10b
python stage10_1_parallel_RDKit_scoring_resumable_training.py --test
python stage10_2_hardware_tuned_batched_AMP_DDP_training.py --test

# Multi-GPU — auto-falls back to CPU/gloo on a 1-GPU box. The smoke test
# covers Stage 9.1; the hardware-tuned best-of-K script uses the same
# rendezvous, batch-splitting and rank-sharding code, so it exercises the
# same path under `torchrun --nproc_per_node=N`.
python stage9_1_ddp_smoketest.py 2

# Not a test, but run it first on any new machine: prints the hardware profile
# and every knob's RESOLVED value, then exits without training.
python stage9_1_batched_GPU_forward_parallel_RDKit_scoring_Lora_finetuning.py --hardware
python stage10_2_hardware_tuned_batched_AMP_DDP_training.py --hardware
```

Stage 9.1's `--test` includes `[0]`, which pins the knob-precedence rule (argument > `config.STAGE9_1_<KNOB>` > `config.STAGE9_1_SPEED`) and asserts that `STAGE9_1_TF32 = False` actually reaches `torch.backends` — the defect that motivated the knobs.

The hardware-tuned best-of-K script's `--test` pins the same precedence rule in its own namespace, and then the claims that make its speedups safe to use:

| check | what it would catch |
|---|---|
| `[1]` unbatched loss `==` the reference implementation's, to zero tolerance | the `"off"` / `"safe"` parity claim silently drifting |
| `[2]` batched forward reproduces the reference with dropout off and `top_k=1` (measured `\|d\| ≈ 5e-07`) | a wrong gather / repeat / scatter index in the batched path |
| `[3]` a molecule inside a padded batch scores the same as alone | padding leaking into real positions |
| `[4]`–`[4c]` shared → one directory, separate → three, variants never merge, a checkpoint written by one stage resumes under another | the shared-lineage flag misrouting output or losing provenance |
| `[5]`–`[6]` interrupt mid-epoch, resume, finish | Adam's moments or the batch cursor not surviving a crash |

All test outputs are written to `{USER_PREFIX}/Output/<stage>/test/` and are wiped at the start of each test run.

---

## 9. Output Reference

### Complete output tree

```
{USER_PREFIX}/Output/
│
├── PLIP_Mask_Calculation/           Stage 1
│   ├── JQ1_A_201.meta.json
│   ├── JQ1_A_201.2d_interactions.png
│   └── ...
│
├── Random_Mask_Calculation/         Stage 1.5
│   ├── JQ1_A_201.meta.random.json
│   └── ...
│
├── masked_smiles_lists/             Stage 2 debug
│   ├── JQ1-A-201_masked_smiles.txt
│   └── ...
│
├── predictions_txt/                 Stage 2 generated molecules
│   └── JQ1-A-201/
│       ├── ia_mask001.txt ... ia_maskN.txt
│       └── rand_mask001.txt ... rand_maskN.txt
│
├── plots/                           Stage 2 incremental plots
│   └── JQ1-A-201_incremental_masking.png
│
├── stage3_analysis/                 Stage 3
│   └── JQ1_A_201/
│       ├── aggregated/
│       │   ├── molecule_grid.png
│       │   ├── histogram_pairwise.png
│       │   └── histogram_vs_original.png
│       └── mask_001/ ... mask_N/
│
├── stage4_br4_matching/             Stage 4
│   └── JQ1_A_201/
│       ├── similarity_score_histogram.png
│       ├── nearest_neighbour_frequency.png
│       └── closeness_summary.txt
│
├── stage5_chembl_matching/          Stage 5
│   ├── chembl_brd4_cache.csv
│   └── JQ1_A_201/
│       ├── similarity_score_histogram.png
│       ├── nearest_neighbour_frequency.png
│       └── closeness_summary.txt
│
├── stage6_docking/                  Stage 6
│   ├── docking_summary.csv
│   └── JQ1_A_201/
│       ├── rec.pdb
│       ├── orig.pdb
│       └── mol_0001/
│           ├── ligand.sdf
│           ├── docked_poses.sdf
│           ├── complex_pose001.pdb
│           └── docking.log
│
└── <property-guided fine-tuning>    Stages 9 / 9a / 10
    │
    │  STAGE10_SHARED_OUTPUT = False (default) — one directory per execution
    │  path, so a timing comparison between them compares execution only:
    │
    ├── stage10_supervised_bestofk/              the reference implementation
    │   └── variant_a/  (and variant_b/)
    │       ├── stage10_checkpoint.json          epoch cursor + history
    │       ├── epoch_001.pt ...                 unfrozen tensors only
    │       ├── config.json, model.safetensors
    │       ├── stage10a_training_curves.png
    │       ├── stage10a_tox_alert_rate.png
    │       ├── stage10a_property_distributions.png
    │       └── stage10a_property_distributions_per_molecule.csv
    │
    ├── stage10_1_supervised_bestofk_parallel/   step-resumable + pooled RDKit
    │   └── variant_a/
    │       ├── stage10_1_checkpoint.pt          + Adam moments + RNG + cursor
    │       ├── stage10_1_checkpoint.json        the same minus tensors
    │       └── ... (as above, named stage10.1a_*)
    │
    ├── stage10_2_supervised_bestofk_hardware_tuned/   batched + AMP + DDP
    │   └── variant_a/
    │       ├── stage10_2_checkpoint.pt
    │       ├── stage10_2_checkpoint.json
    │       └── ... (as above, named stage10.2a_*)
    │
    │  STAGE10_SHARED_OUTPUT = True — the three above are replaced by ONE
    │  directory holding ONE continuable checkpoint and ONE lineage model:
    │
    └── stage10_family_shared/
        └── variant_a/
            ├── stage10_lineage.pt       weights + Adam + history + cursor
            │                            + RNG + provenance
            ├── stage10_lineage.json     readable with `cat`; says which stage
            │                            trained which epochs, at what precision
            ├── epoch_001.pt ...         per-epoch snapshots, whoever wrote them
            ├── config.json, model.safetensors     the lineage endpoint
            ├── stage10a_training_curves.png       per-stage figures survive
            ├── stage10a_tox_alert_rate.png
            ├── stage10.1a_training_curves.png
            ├── stage10.1a_tox_alert_rate.png
            ├── stage10.2a_training_curves.png
            └── stage10.2a_tox_alert_rate.png
```

---

## 10. Example Walkthrough

The following walkthrough uses the BRD4–JQ1 co-crystal structure `3MXF` as the input. JQ1 is the reference BRD4 inhibitor with IC₅₀ ≈ 77 nM [3].

### Step 1 — Download input data

```python
import requests, os

# Download PDB structure
pdb_id = "3MXF"
url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
os.makedirs("Dummy_data/PDB", exist_ok=True)
with open(f"Dummy_data/PDB/{pdb_id}.pdb", "w") as f:
    f.write(requests.get(url).text)
print(f"Downloaded {pdb_id}.pdb")
```

Then generate the PLIP XML via the [PLIP web server](https://plip-tool.biotec.tu-dresden.de/plip-web/plip/index) or the local PLIP tool and place it in `Dummy_data/plip/3MXF.xml`.

### Step 2 — Configure

Edit `config.py`:
```python
BASE_DIR    = "/content/drive/MyDrive/GenAI4Drug"
USER_PREFIX = "MyName"

PIPELINE_INPUTS = [
    {
        "pdb_path":      BASE_PDB_PATH + "3MXF",
        "plip_xml_path": BASE_XML_PATH + "3MXF",
        "resname": "JQ1",
        "chain":   "A",
        "resseq":  1,
    },
]
```

### Step 3 — Run the pipeline

```bash
python stage1_mask_calculation.py
# → JQ1 has N interaction-aware atoms identified by PLIP
# → JQ1_A_1.meta.json written

python stage1_5_random_masking.py
# → N random indices sampled (seed=42)
# → JQ1_A_1.meta.random.json written

python stage2_molecule_generation.py
# → ChemBERTa generates molecules for mask_count = 1…N
# → Plots: IA vs random unique valid SMILES count

python stage3_analysis.py
# → Molecule grids and Tanimoto histograms

python stage4_br4_matching.py
# → Nearest BR4 reference ligand for each generated molecule
# → Closeness summary: "X% are most similar to JQ1"

python stage5_chembl_matching.py
# → Same analysis against full ChEMBL BRD4 active set

python stage6_docking.py
# → GNINA docks all generated molecules into the BRD4 pocket
# → docking_summary.csv with CNNscore, CNNaffinity, Vinardo scores
```

### Step 4 — Identify top candidates

```python
import pandas as pd

df = pd.read_csv("Output/stage6_docking/docking_summary.csv")

# Keep best pose per molecule, sort by CNN affinity
best = (df.sort_values("CNNaffinity", ascending=False)
          .groupby(["ligand_group", "mol_idx"])
          .first()
          .reset_index())

# Show top 10
print(best[["ligand_group", "smiles", "CNNscore",
            "CNNaffinity", "minimizedAffinity"]].head(10))
```

---

## 11. Scientific References

[1] Shi, J., & Vakoc, C. R. (2014). The mechanisms behind the therapeutic activity of BET bromodomain inhibition. *Molecular Cell*, 54(5), 728–736. https://doi.org/10.1016/j.molcel.2014.05.016

[2] Filippakopoulos, P., & Knapp, S. (2014). Targeting bromodomains: epigenetic readers of lysine acetylation. *Nature Reviews Drug Discovery*, 13(5), 337–356. https://doi.org/10.1038/nrd4286

[3] Filippakopoulos, P., et al. (2010). Selective inhibition of BET bromodomains. *Nature*, 468(7327), 1067–1073. https://doi.org/10.1038/nature09504

[4] Chithrananda, S., Grand, G., & Ramsundar, B. (2020). ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction. *arXiv preprint* arXiv:2010.09885. https://arxiv.org/abs/2010.09885

[5] Salentin, S., Schreiber, S., Haupt, V. J., Adasme, M. F., & Schroeder, M. (2015). PLIP: fully automated protein–ligand interaction profiler. *Nucleic Acids Research*, 43(W1), W443–W447. https://doi.org/10.1093/nar/gkv315

[6] Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742–754. https://doi.org/10.1021/ci100050t

[7] McNutt, A. T., et al. (2021). GNINA 1.0: molecular docking with deep learning. *Journal of Cheminformatics*, 13(1), 43. https://doi.org/10.1186/s13321-021-00522-2

[8] Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org

[9] Krenn, M., Häse, F., Nigam, A., Friederich, P., & Aspuru-Guzik, A. (2020). Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation. *Machine Learning: Science and Technology*, 1(4), 045024. https://doi.org/10.1088/2632-2153/aba947

[10] Gaulton, A., et al. (2017). The ChEMBL database in 2017. *Nucleic Acids Research*, 45(D1), D945–D954. https://doi.org/10.1093/nar/gkw1074

---

## 12. Acknowledgements

- **PLIP** (Protein–Ligand Interaction Profiler) — TU Dresden, Schroeder lab
- **ChemBERTa** — Bharath Ramsundar lab (DeepChem)
- **GNINA** — David Koes lab, University of Pittsburgh
- **RDKit** — Greg Landrum and the RDKit community
- **ChEMBL** — EMBL-EBI

---

## Licence

This project is released for academic and research use. Please cite the relevant tools (see [Scientific References](#11-scientific-references)) in any publication using this pipeline.

---

*Pipeline developed as part of a structure-guided generative drug design project targeting BRD4.*

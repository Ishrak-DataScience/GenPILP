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
```

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
└── stage6_docking/                  Stage 6
    ├── docking_summary.csv
    └── JQ1_A_201/
        ├── rec.pdb
        ├── orig.pdb
        └── mol_0001/
            ├── ligand.sdf
            ├── docked_poses.sdf
            ├── complex_pose001.pdb
            └── docking.log
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

# -*- coding: utf-8 -*-
"""
stage2_molecule_generation.py
==============================
Stage 2 of the pipeline:

    Stage-1 JSONs  (interaction-aware masks)   ┐
                                                ├─→ ChemBERTa → plots + SMILES txt
    Stage-1.5 JSONs (random masks)             ┘

For each ligand, the stage iterates mask_count = 1, 2, … N where
N = len(stage1_meta["masked_atom_indices"]).

At each mask_count:
  • Interaction-aware masked SMILES: first `mask_count` indices from the
    Stage-1 ordered list  →  ChemBERTa  →  count unique valid SMILES
  • Random masked SMILES:            first `mask_count` indices from the
    Stage-1.5 ordered list →  ChemBERTa  →  count unique valid SMILES

Output per ligand:
  • One PNG plot saved to config.PLOT_DIR
  • One txt file per (ligand, mask_count, strategy) saved to config.PRED_DIR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS (read before use)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1. "First mask_count indices" for the interaction-aware line means the
    literal first mask_count elements of stage1["masked_atom_indices"]
    in the order Stage 1 stored them (PLIP parsing order, not ranked
    by importance). (Ambiguity 2 → Option A)

A2. "First mask_count indices" for the random line means the literal
    first mask_count elements of stage1_5["masked_atom_indices"], which
    were sampled once with a fixed seed in Stage 1.5.

A3. Both strategy lists have the same length N per ligand (enforced by
    Stage 1.5). The x-axis runs from 1 to N for each ligand's own plot.
    (Ambiguity 3 → Option A)

A4. Stage-1 and Stage-1.5 JSONs are matched by the ligand key string
    "{resname}-{chain}-{resseq}". If no Stage-1.5 match is found for a
    ligand, that ligand is skipped with a warning.

A5. If Stage-1 and Stage-1.5 lists have different lengths (should not
    happen if Stage 1.5 ran correctly), the shorter length is used and
    a warning is printed.

A6. INCREMENTAL_NUM_SAMPLES ChemBERTa samples are drawn per
    (ligand, mask_count, strategy) cell — NOT averaged over multiple
    random subsets. (Ambiguity 2 → Option A)

A7. A SMILES candidate is considered valid if and only if
    RDKit's Chem.MolFromSmiles() returns a non-None object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1:  python stage1_mask_calculation.py
Step 2:  python stage1_5_random_masking.py
Step 3:  python stage2_molecule_generation.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO TEST (without real PDB / PLIP files)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run the self-contained test at the bottom of this file:
    python stage2_molecule_generation.py --test

The test:
  1. Creates synthetic Stage-1 and Stage-1.5 JSON files for two fake
     ligands (aspirin and caffeine).
  2. Loads ChemBERTa and runs the full incremental loop with
     num_samples=5 (fast, not publication quality).
  3. Checks that plots are created and that all result dicts are
     non-empty.
  4. Prints PASS / FAIL for each assertion.
"""

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from transformers import AutoModelForMaskedLM, AutoTokenizer

import config
from stage1_5_random_masking import ligand_key
from stage1_mask_calculation import mask_atoms_in_smiles


# ════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_chemberta(model_name: str = config.CHEMBERTA_MODEL):
    """Load ChemBERTa tokenizer + model once. Returns (tokenizer, model, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer.mask_token == "<mask>", (
        f"Unexpected mask token '{tokenizer.mask_token}'. "
        "Make sure you are using seyonec/ChemBERTa-zinc-base-v1."
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    print("✅ ChemBERTa loaded.")
    return tokenizer, model, device


# ════════════════════════════════════════════════════════════════════════════
#  SEQUENTIAL ChemBERTa GENERATION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_smiles_sequential(
    smiles_masked: str,
    tokenizer,
    model,
    device: str,
    top_k: int = config.TOP_K,
    num_samples: int = config.INCREMENTAL_NUM_SAMPLES,
    temperature: float = config.TEMPERATURE,
    save_path: Optional[str] = None,
) -> List[str]:
    """
    Sequentially fill all <mask> tokens in smiles_masked left-to-right.
    Each fill is conditioned on all previously filled tokens.

    Returns a sorted list of unique valid SMILES.
    Optionally saves them to save_path (one per line).
    """
    max_len = getattr(tokenizer, "model_max_length", 512)
    if max_len is None or max_len > 1024:
        max_len = 512

    enc = tokenizer(
        smiles_masked, return_tensors="pt", truncation=True, max_length=max_len
    ).to(device)
    base_ids  = enc["input_ids"][0]
    base_attn = enc["attention_mask"][0]
    mask_id   = tokenizer.mask_token_id

    mask_positions = (base_ids == mask_id).nonzero(as_tuple=True)[0].tolist()
    if not mask_positions:
        # Input has no masks — just validate the literal string
        clean = smiles_masked.replace(" ", "")
        if Chem.MolFromSmiles(clean) is not None:
            return [clean]
        return []

    valid_set: set = set()
    invalid   = 0

    for _ in range(num_samples):
        ids = base_ids.clone()
        for pos in mask_positions:
            out     = model(input_ids=ids.unsqueeze(0), attention_mask=base_attn.unsqueeze(0))
            logits  = out.logits[0, pos] / max(temperature, 1e-8)
            probs   = torch.softmax(logits, dim=-1)
            topk    = torch.topk(probs, k=top_k)
            top_p   = topk.values / topk.values.sum()
            sampled = torch.multinomial(top_p, num_samples=1).item()
            ids[pos] = topk.indices[sampled]

        cand = tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
        if Chem.MolFromSmiles(cand) is not None:
            valid_set.add(cand)
        else:
            invalid += 1

    valid_list = sorted(valid_set)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(valid_list) + "\n")

    return valid_list


# ════════════════════════════════════════════════════════════════════════════
#  JSON LOADING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_json_folder(folder: str) -> Dict[str, dict]:
    """
    Load all *.json files in folder.
    Returns {ligand_key: meta_dict}.
    Files are sorted by name so behaviour is deterministic when duplicates exist.
    """
    result: Dict[str, dict] = {}
    paths = sorted(glob.glob(os.path.join(folder, "*.json")))
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                meta = json.load(f)
            key = ligand_key(meta)
            if key in result:
                print(f"  ⚠️  Duplicate ligand key '{key}' in {folder}. "
                      f"Overwriting with {os.path.basename(p)}.")
            result[key] = meta
        except Exception as e:
            print(f"  ⚠️  Could not load {p}: {e}")
    return result


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


# ════════════════════════════════════════════════════════════════════════════
#  INCREMENTAL MASKING LOOP  (core of Stage 2)
# ════════════════════════════════════════════════════════════════════════════

def run_incremental_for_ligand(
    lig_id: str,
    smiles: str,
    ia_indices: List[int],
    rand_indices: List[int],
    tokenizer,
    model,
    device: str,
    pred_dir: str,
    num_samples: int = config.INCREMENTAL_NUM_SAMPLES,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Run the incremental masking loop for a single ligand.

    For mask_count in 1 … N:
      - Build masked SMILES using ia_indices[:mask_count]    (interaction-aware)
      - Build masked SMILES using rand_indices[:mask_count]  (random)
      - Generate SMILES from each, record unique valid count

    Returns
    -------
    mask_counts  : [1, 2, …, N]
    ia_valids    : unique valid SMILES count at each mask_count  (IA strategy)
    rand_valids  : unique valid SMILES count at each mask_count  (random strategy)
    """
    N = min(len(ia_indices), len(rand_indices))   # Assumption A5
    if len(ia_indices) != len(rand_indices):
        print(
            f"  ⚠️  {lig_id}: IA list length ({len(ia_indices)}) ≠ "
            f"random list length ({len(rand_indices)}). Using N={N}."
        )

    mask_counts: List[int] = []
    ia_valids:   List[int] = []
    rand_valids: List[int] = []

    lig_pred_dir = os.path.join(pred_dir, safe_filename(lig_id))
    os.makedirs(lig_pred_dir, exist_ok=True)

    for mask_count in range(1, N + 1):
        print(f"    mask_count = {mask_count:>3d}/{N}", end="  ", flush=True)

        # ── Interaction-aware masked SMILES ───────────────────────────────────
        ia_masked = mask_atoms_in_smiles(smiles, ia_indices[:mask_count], tokenizer)
        ia_save   = os.path.join(lig_pred_dir, f"ia_mask{mask_count:03d}.txt")
        ia_preds  = generate_smiles_sequential(
            smiles_masked = ia_masked,
            tokenizer     = tokenizer,
            model         = model,
            device        = device,
            num_samples   = num_samples,
            save_path     = ia_save,
        )
        n_ia = len(ia_preds)

        # ── Random masked SMILES ──────────────────────────────────────────────
        rand_masked = mask_atoms_in_smiles(smiles, rand_indices[:mask_count], tokenizer)
        rand_save   = os.path.join(lig_pred_dir, f"rand_mask{mask_count:03d}.txt")
        rand_preds  = generate_smiles_sequential(
            smiles_masked = rand_masked,
            tokenizer     = tokenizer,
            model         = model,
            device        = device,
            num_samples   = num_samples,
            save_path     = rand_save,
        )
        n_rand = len(rand_preds)

        print(f"IA valid = {n_ia:>4d}   rand valid = {n_rand:>4d}")

        mask_counts.append(mask_count)
        ia_valids.append(n_ia)
        rand_valids.append(n_rand)

    return mask_counts, ia_valids, rand_valids


# ════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_incremental_results(
    lig_id: str,
    mask_counts: List[int],
    ia_valids: List[int],
    rand_valids: List[int],
    plot_dir: str,
    num_samples: int,
) -> str:
    """
    Save a PNG showing interaction-aware vs random unique valid SMILES
    as a function of mask count. Returns the saved file path.
    """
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{safe_filename(lig_id)}_incremental_masking.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        mask_counts, ia_valids,
        marker="o", linewidth=2, color="#1f77b4",
        label="Interaction-Aware (Stage 1)",
    )
    ax.plot(
        mask_counts, rand_valids,
        marker="s", linewidth=2, color="#ff7f0e", linestyle="--",
        label="Random (Stage 1.5)",
    )

    # Annotate the last point of each line
    if mask_counts:
        ax.annotate(
            str(ia_valids[-1]),
            xy=(mask_counts[-1], ia_valids[-1]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=9, color="#1f77b4",
        )
        ax.annotate(
            str(rand_valids[-1]),
            xy=(mask_counts[-1], rand_valids[-1]),
            xytext=(5, -13), textcoords="offset points",
            fontsize=9, color="#ff7f0e",
        )

    ax.set_xlabel("Number of Masks", fontsize=13)
    ax.set_ylabel("Unique Valid SMILES Generated", fontsize=13)
    ax.set_title(
        f"Incremental Masking: {lig_id}\n"
        f"(ChemBERTa samples per cell = {num_samples})",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    if mask_counts and mask_counts[-1] <= 30:
        ax.set_xticks(mask_counts)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼  Plot saved: {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
#  MAIN GENERATION FUNCTION  (importable)
# ════════════════════════════════════════════════════════════════════════════

def run_generation(
    stage1_dir:  Optional[str] = None,
    stage15_dir: Optional[str] = None,
    pred_dir:    Optional[str] = None,
    plot_dir:    Optional[str] = None,
    num_samples: int = config.INCREMENTAL_NUM_SAMPLES,
) -> Dict[str, dict]:
    """
    Main Stage-2 function.

    Returns
    -------
    results : dict keyed by ligand_id, each value:
        {
          "mask_counts":  [1, 2, …, N],
          "ia_valids":    [int, …],
          "rand_valids":  [int, …],
          "plot_path":    str,
        }
    """
    stage1_dir  = stage1_dir  or config.MASK_CALC_OUTDIR
    stage15_dir = stage15_dir or config.RANDOM_MASK_OUTDIR
    pred_dir    = pred_dir    or config.PRED_DIR
    plot_dir    = plot_dir    or config.PLOT_DIR

    for d in [pred_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    stage1_metas  = load_json_folder(stage1_dir)
    stage15_metas = load_json_folder(stage15_dir)

    print(f"\n  Stage-1   JSONs loaded:  {len(stage1_metas)}")
    print(f"  Stage-1.5 JSONs loaded:  {len(stage15_metas)}")

    matched       = sorted(set(stage1_metas) & set(stage15_metas))
    unmatched_s1  = sorted(set(stage1_metas)  - set(stage15_metas))
    unmatched_s15 = sorted(set(stage15_metas) - set(stage1_metas))

    if unmatched_s1:
        print(f"\n  ⚠️  Stage-1 ligands with NO Stage-1.5 match (skipped): {unmatched_s1}")
        print("     Did you run stage1_5_random_masking.py?")
    if unmatched_s15:
        print(f"\n  ⚠️  Stage-1.5 ligands with NO Stage-1 match (ignored): {unmatched_s15}")

    if not matched:
        print("\n  ❌ No matched ligands found.  Exiting.")
        return {}

    tokenizer, model, device = load_chemberta()

    all_results: Dict[str, dict] = {}

    for lig_id in matched:
        s1  = stage1_metas[lig_id]
        s15 = stage15_metas[lig_id]

        smiles       = s1["smiles"]
        ia_indices   = s1["masked_atom_indices"]
        rand_indices = s15["masked_atom_indices"]

        if not ia_indices:
            print(f"\n  ⚠️  {lig_id}: empty ia_indices, skipping.")
            continue

        N = min(len(ia_indices), len(rand_indices))
        print(f"\n{'='*60}")
        print(f"Ligand : {lig_id}")
        print(f"SMILES : {smiles}")
        print(f"N      : {N}  (interaction-aware indices: {ia_indices})")
        print(f"{'='*60}")

        mask_counts, ia_valids, rand_valids = run_incremental_for_ligand(
            lig_id       = lig_id,
            smiles       = smiles,
            ia_indices   = ia_indices,
            rand_indices = rand_indices,
            tokenizer    = tokenizer,
            model        = model,
            device       = device,
            pred_dir     = pred_dir,
            num_samples  = num_samples,
        )

        plot_path = plot_incremental_results(
            lig_id      = lig_id,
            mask_counts = mask_counts,
            ia_valids   = ia_valids,
            rand_valids = rand_valids,
            plot_dir    = plot_dir,
            num_samples = num_samples,
        )

        all_results[lig_id] = {
            "mask_counts":  mask_counts,
            "ia_valids":    ia_valids,
            "rand_valids":  rand_valids,
            "plot_path":    plot_path,
        }

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED TEST  (python stage2_molecule_generation.py --test)
# ════════════════════════════════════════════════════════════════════════════

def _run_test():
    """
    Smoke-test that does NOT require PDB / PLIP files.
    Uses aspirin and caffeine as synthetic ligands.
    Runs with num_samples=5 (finishes in ~2 min on CPU).
    """
    print("\n" + "=" * 60)
    print("STAGE 2 SELF-TEST  (num_samples=5, 2 synthetic ligands)")
    print("=" * 60)

    test_ligands = [
        {
            "smiles":      "CC(=O)Oc1ccccc1C(=O)O",       # aspirin, 13 heavy atoms
            "ligand":      {"resname": "ASP", "chain": "A", "resseq": 1},
            "ia_indices":  [0, 3, 6, 9],
            "rand_indices": [1, 4, 7, 10],
        },
        {
            "smiles":      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine, 14 heavy atoms
            "ligand":      {"resname": "CAF", "chain": "A", "resseq": 2},
            "ia_indices":  [0, 2, 5, 8, 11],
            "rand_indices": [1, 3, 6, 9, 12],
        },
    ]

    passed = 0
    failed = 0

    def check(condition, msg):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ PASS  {msg}")
            passed += 1
        else:
            print(f"  ❌ FAIL  {msg}")
            failed += 1

    # ── Wipe and recreate TEST_DIR sub-folders (Option A) ────────────────────
    s1_dir   = os.path.join(config.TEST_DIR, "stage1")
    s15_dir  = os.path.join(config.TEST_DIR, "stage15")
    pred_dir = os.path.join(config.TEST_DIR, "preds")
    plot_dir = os.path.join(config.TEST_DIR, "plots")

    if os.path.exists(config.TEST_DIR):
        shutil.rmtree(config.TEST_DIR)
        print(f"  🗑  Wiped existing TEST_DIR: {config.TEST_DIR}")

    for d in [s1_dir, s15_dir, pred_dir, plot_dir]:
        os.makedirs(d)
    print(f"  📁 Created fresh TEST_DIR:  {config.TEST_DIR}\n")

    for lig in test_ligands:
        key = (f"{lig['ligand']['resname']}-"
               f"{lig['ligand']['chain']}-"
               f"{lig['ligand']['resseq']}")
        with open(os.path.join(s1_dir,  key + ".meta.json"),  "w") as f:
            json.dump({"smiles": lig["smiles"], "ligand": lig["ligand"],
                       "masked_atom_indices": lig["ia_indices"],
                       "masking_mode": "attractive"}, f)
        with open(os.path.join(s15_dir, key + ".random.json"), "w") as f:
            json.dump({"smiles": lig["smiles"], "ligand": lig["ligand"],
                       "masked_atom_indices": lig["rand_indices"],
                       "masking_mode": "random", "source": "stage1_5"}, f)

    results = run_generation(
        stage1_dir  = s1_dir,
        stage15_dir = s15_dir,
        pred_dir    = pred_dir,
        plot_dir    = plot_dir,
        num_samples = 5,
    )

    check(len(results) == 2, f"Expected 2 ligands, got {len(results)}")

    for lig in test_ligands:
        key = (f"{lig['ligand']['resname']}-"
               f"{lig['ligand']['chain']}-"
               f"{lig['ligand']['resseq']}")
        if key not in results:
            check(False, f"{key} missing from results"); continue

        r = results[key]
        N = len(lig["ia_indices"])

        check(r["mask_counts"] == list(range(1, N + 1)),
              f"{key}: mask_counts = [1..{N}]")
        check(len(r["ia_valids"]) == N,
              f"{key}: ia_valids has {N} entries")
        check(len(r["rand_valids"]) == N,
              f"{key}: rand_valids has {N} entries")
        check(os.path.exists(r["plot_path"]),
              f"{key}: PNG exists at {r['plot_path']}")
        check(all(isinstance(v, int) and v >= 0 for v in r["ia_valids"]),
              f"{key}: all ia_valids ≥ 0")
        check(all(isinstance(v, int) and v >= 0 for v in r["rand_valids"]),
              f"{key}: all rand_valids ≥ 0")

        lig_pred = os.path.join(pred_dir, safe_filename(key))
        for mc in range(1, N + 1):
            check(os.path.exists(os.path.join(lig_pred, f"ia_mask{mc:03d}.txt")),
                  f"{key}: ia_mask{mc:03d}.txt exists")
            check(os.path.exists(os.path.join(lig_pred, f"rand_mask{mc:03d}.txt")),
                  f"{key}: rand_mask{mc:03d}.txt exists")

    print(f"\n{'='*60}")
    print(f"Test complete:  {passed} passed,  {failed} failed.")
    print("✅ All tests passed." if failed == 0 else "❌ Some tests failed.")
    return failed == 0


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Incremental Molecule Generation")
    parser.add_argument("--test", action="store_true",
                        help="Run self-contained smoke test (no PDB/PLIP files needed).")
    args = parser.parse_args()

    if args.test:
        ok = _run_test()
        sys.exit(0 if ok else 1)

    print("\n" + "=" * 60)
    print("STAGE 2: INCREMENTAL MOLECULE GENERATION")
    print("=" * 60)

    results = run_generation()

    print("\n" + "=" * 60)
    print("✅ Stage 2 complete.")
    print(f"   Plots saved to:       {config.PLOT_DIR}")
    print(f"   Predictions saved to: {config.PRED_DIR}")
    print(f"   Ligands processed:    {len(results)}")
    for lig_id, r in sorted(results.items()):
        n = len(r["mask_counts"])
        print(f"   • {lig_id}  N={n}  "
              f"peak_ia={max(r['ia_valids']) if r['ia_valids'] else 0}  "
              f"peak_rand={max(r['rand_valids']) if r['rand_valids'] else 0}")


if __name__ == "__main__":
    main()

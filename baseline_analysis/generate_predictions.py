# -*- coding: utf-8 -*-
"""
generate_predictions.py
=======================
Step 3: fill every frozen mask with VANILLA ChemBERTa and measure what came out.

Reads masks.csv (produced by build_masks.py, never re-derives a mask), runs one
MLM forward pass per row, and writes predictions.csv -- one row per
(complex, arm, k) with the generated SMILES plus its RDKit property profile.

WHY THE DECODING IS BORROWED, NOT WRITTEN HERE
-----------------------------------------------
Generation calls stage9_masked_property_finetune.generate_completion, which
wraps reinforce_rollout_oneshot: a SINGLE forward pass, every <mask> sampled
independently from it (top-k + temperature). That is the same decoder Stage 9
trains with and Stage 9a baselines with, so a baseline number here is directly
comparable to those figures -- the only difference is the weights. Properties
come from the same module's compute_property_components, so QED / SA / novelty /
PAINS / Brenk / Tox21 mean exactly what they mean everywhere else in the
pipeline. Tox21 is filled in only when config.STAGE9_TOX21_MODEL_DIR points at a
real checkpoint; otherwise the column is empty and the report says so.

REPRODUCIBILITY
----------------
Sampling is seeded per row from masks.csv's gen_seed (torch.manual_seed before
each rollout), so a re-run reproduces the same molecules. Before generating,
masks.csv's sha256 is checked against the manifest written by build_masks.py:
if the masks changed, the run stops rather than quietly mixing inputs. Pointing
--model at a different checkpoint reuses the identical masks and records the new
model in the manifest -- the "reproduce with a different model" path.

Resumable: rows already in predictions.csv are kept and skipped, so an
interrupted run continues where it stopped. --overwrite starts clean.

HOW TO RUN
-----------
    python baseline_analysis/generate_predictions.py --out-dir "$BASELINE_DIR"
    python baseline_analysis/generate_predictions.py --model <other/checkpoint>
    python baseline_analysis/generate_predictions.py --test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_common as bc  # noqa: E402

PRED_FIELDS = [
    "complex_id", "pdb_id", "resname", "chain", "resseq", "uniprot", "fda_status",
    "arm", "k", "mask_seed", "gen_seed", "model",
    "parent_smiles", "masked_smiles", "generated_smiles",
    "n_mask_tokens", "mask_token_frac", "identical_to_parent",
    "valid", "qed", "sa_raw", "sa_norm", "novelty",
    "pains", "brenk", "any_alert", "n_alerts", "alert_free", "tox21",
    "mw", "logp", "hbd", "hba", "rotb", "rings", "heavy_atoms", "lipinski_violations",
]

# Properties compute_property_components does not cover but "drug-like?" needs.
t0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def rdkit_druglikeness(smiles: str) -> Dict[str, Optional[float]]:
    """
    Lipinski-style descriptors for the drug-likeness question. QED and SA come
    from Stage 9's scorer; these are the raw numbers a reader expects to see
    next to them. Returns Nones for an unparseable molecule rather than zeros.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    out: Dict[str, Optional[float]] = {k: None for k in
                                       ("mw", "logp", "hbd", "hba", "rotb", "rings",
                                        "heavy_atoms", "lipinski_violations")}
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return out
    mw   = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd  = int(rdMolDescriptors.CalcNumHBD(mol))
    hba  = int(rdMolDescriptors.CalcNumHBA(mol))
    out.update(
        mw=round(mw, 2), logp=round(logp, 3), hbd=hbd, hba=hba,
        rotb=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        rings=int(rdMolDescriptors.CalcNumRings(mol)),
        heavy_atoms=int(mol.GetNumHeavyAtoms()),
        lipinski_violations=int((mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)),
    )
    return out


def canonical(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else ""


def verify_masks_unchanged(root: str, masks_csv: str, strict: bool = True) -> Optional[str]:
    """
    Compare masks.csv's sha256 against what build_masks.py recorded. Returns the
    digest. Raises when they differ and strict -- generating against silently
    edited masks would break the one guarantee this bundle is built on.
    """
    digest = bc.sha256_file(masks_csv)
    manifest_path = bc.paths(root)["manifest"]
    recorded = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                recorded = (json.load(f).get("masks") or {}).get("masks_csv_sha256")
        except (json.JSONDecodeError, OSError):
            recorded = None
    if recorded and recorded != digest:
        msg = (f"masks.csv has changed since build_masks.py ran\n"
               f"  recorded sha256 : {recorded}\n"
               f"  current  sha256 : {digest}\n"
               f"Re-run build_masks.py, or pass --allow-mask-drift to proceed anyway.")
        if strict:
            raise SystemExit("ERROR: " + msg)
        print("  ⚠️  " + msg)
    elif not recorded:
        print("  (no recorded mask hash in manifest.json; nothing to compare against)")
    return digest


def run(args: argparse.Namespace) -> int:
    import torch

    root = bc.bundle_dir(args.out_dir)
    P = bc.paths(root)
    if not os.path.exists(P["masks"]):
        print(f"ERROR: no masks.csv at {P['masks']}. Run build_masks.py first.")
        return 2

    digest = verify_masks_unchanged(root, P["masks"], strict=not args.allow_mask_drift)
    rows = bc.read_csv_rows(P["masks"])
    usable = [r for r in rows if not r.get("skip_reason")]
    if args.limit:
        usable = usable[: args.limit]
    log(f"masks.csv: {len(rows)} rows, {len(usable)} usable")

    done: Dict[tuple, dict] = {}
    if os.path.exists(P["predictions"]) and not args.overwrite:
        for r in bc.read_csv_rows(P["predictions"]):
            done[(r["complex_id"], r["arm"], str(r["k"]))] = r
        log(f"resuming: {len(done)} predictions already on disk")

    todo = [r for r in usable if (r["complex_id"], r["arm"], str(r["k"])) not in done]
    log(f"to generate: {len(todo)}")

    settings = bc.baseline_settings()
    model_name = args.model or settings["chemberta_model"]
    out_rows: List[dict] = list(done.values())

    if todo:
        import stage9_masked_property_finetune as s9
        from transformers import AutoModelForMaskedLM

        tokenizer = s9.get_chemberta_tokenizer(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"loading {model_name} on {device} (vanilla, no adapter)")
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()

        top_k = int(args.top_k if args.top_k is not None else settings["top_k"])
        temperature = float(args.temperature if args.temperature is not None
                            else settings["temperature"])

        try:
            from tqdm.auto import tqdm
            it = tqdm(todo, desc="generating", unit="mol")
        except Exception:
            it = todo

        for i, r in enumerate(it, 1):
            torch.manual_seed(int(r["gen_seed"]))      # per-row, so the run replays
            if device == "cuda":
                torch.cuda.manual_seed_all(int(r["gen_seed"]))
            gen = s9.generate_completion(
                r["masked_smiles"], tokenizer, model, device,
                top_k=top_k, temperature=temperature,
            )
            parent = r["parent_smiles"]
            props = s9.compute_property_components(gen, parent)
            drug = rdkit_druglikeness(gen)
            cg, cp = canonical(gen), canonical(parent)
            out_rows.append({
                **{k: r.get(k, "") for k in
                   ("complex_id", "pdb_id", "resname", "chain", "resseq", "uniprot",
                    "fda_status", "arm", "k", "mask_seed", "gen_seed",
                    "parent_smiles", "masked_smiles", "n_mask_tokens", "mask_token_frac")},
                "model": model_name,
                "generated_smiles": gen,
                "identical_to_parent": int(bool(cg) and cg == cp),
                **{k: ("" if props.get(k) is None else props[k]) for k in s9.PROPERTY_KEYS},
                **{k: ("" if v is None else v) for k, v in drug.items()},
            })
            if i % max(1, args.flush_every) == 0:
                bc.write_csv_rows(P["predictions"], out_rows, PRED_FIELDS)

    bc.write_csv_rows(P["predictions"], out_rows, PRED_FIELDS)

    valid = [r for r in out_rows if str(r.get("valid")) in ("1", "1.0")]
    by_arm = {a: [r for r in out_rows if r["arm"] == a] for a in bc.ARMS}
    bc.write_manifest(root, "generation", {
        "model": model_name,
        "masks_csv_sha256": digest,
        "predictions_sha256": bc.sha256_file(P["predictions"]),
        "n_rows": len(out_rows), "n_valid": len(valid),
        "top_k": int(args.top_k if args.top_k is not None else settings["top_k"]),
        "temperature": float(args.temperature if args.temperature is not None
                             else settings["temperature"]),
        "tox21_checkpoint": getattr(bc.config, "STAGE9_TOX21_MODEL_DIR", "") or None,
        "settings": settings,
    })

    print("\n── generation summary ────────────────────────────────────────")
    print(f"{'arm':<10}{'n':>6}{'valid':>8}{'valid%':>9}{'=parent':>9}{'mean QED':>10}{'mean nov':>10}")
    for a in bc.ARMS:
        rs = by_arm[a]
        if not rs:
            continue
        v = [r for r in rs if str(r.get("valid")) in ("1", "1.0")]
        qed = [float(r["qed"]) for r in v if r.get("qed") not in ("", None)]
        nov = [float(r["novelty"]) for r in v if r.get("novelty") not in ("", None)]
        same = sum(1 for r in rs if str(r.get("identical_to_parent")) == "1")
        print(f"{a:<10}{len(rs):>6}{len(v):>8}{100*len(v)/len(rs):>8.1f}%{same:>9}"
              f"{(sum(qed)/len(qed) if qed else float('nan')):>10.3f}"
              f"{(sum(nov)/len(nov) if nov else float('nan')):>10.3f}")
    print(f"\npredictions.csv: {P['predictions']}")
    return 0


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST  (no network, no real checkpoint)
# ════════════════════════════════════════════════════════════════════════════

def _self_test() -> int:
    import tempfile

    print("generate_predictions.py self-test")

    # Drug-likeness on a molecule with known descriptors.
    d = rdkit_druglikeness("CC(=O)Oc1ccccc1C(=O)O")          # aspirin
    assert d["heavy_atoms"] == 13 and 179 < d["mw"] < 181, d
    assert d["lipinski_violations"] == 0, d
    print(f"  ✓ druglikeness: aspirin MW={d['mw']} logP={d['logp']} "
          f"HBD={d['hbd']} HBA={d['hba']} Lipinski violations={d['lipinski_violations']}")
    big = rdkit_druglikeness("C" * 60)
    assert big["lipinski_violations"] >= 2, big
    assert rdkit_druglikeness("not-a-molecule")["mw"] is None
    print("  ✓ druglikeness: oversized chain flags Lipinski; invalid SMILES -> None, not 0")

    assert canonical("C1=CC=CC=C1") == canonical("c1ccccc1")
    assert canonical("nope") == ""
    print("  ✓ canonical(): resonance forms agree, invalid -> ''")

    # Property scoring wiring (real Stage 9 scorer, no model needed).
    import stage9_masked_property_finetune as s9
    p = s9.compute_property_components("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O")
    assert p["valid"] == 1.0 and p["novelty"] == 0.0
    inv = s9.compute_property_components("xx", "CC(=O)Oc1ccccc1C(=O)O")
    assert inv["valid"] == 0.0 and inv["qed"] is None
    print("  ✓ property scorer: identical parent -> novelty 0; invalid -> valid=0, QED None")

    # Seeded decoding is reproducible, with a stub model (no HF download).
    import torch

    class _StubOut:
        def __init__(self, logits):
            self.logits = logits

    class _StubModel:
        """Returns fixed pseudo-random logits so sampling is the only variable."""
        def __init__(self, vocab):
            g = torch.Generator().manual_seed(7)
            self.table = torch.randn(64, vocab, generator=g)

        def __call__(self, input_ids=None, attention_mask=None, **kw):
            n = input_ids.shape[1]
            return _StubOut(self.table[:n].unsqueeze(0))

    tok = s9.get_chemberta_tokenizer()
    stub = _StubModel(len(tok))
    masked = "CC(=O)O" + tok.mask_token + "c1ccccc1"

    def _gen(seed):
        torch.manual_seed(seed)
        return s9.generate_completion(masked, tok, stub, "cpu", top_k=20, temperature=1.2)

    a1, a2, b = _gen(1000), _gen(1000), _gen(1001)
    assert a1 == a2, (a1, a2)
    print(f"  ✓ seeded decoding: same seed -> identical output ({a1[:34]!r}...)")
    print(f"  {'✓' if a1 != b else '!'} different seed -> "
          f"{'different output' if a1 != b else 'same output (possible, small vocab draw)'}")

    # Resume: existing rows are kept and not regenerated.
    with tempfile.TemporaryDirectory() as td:
        P = bc.paths(td)
        mask_rows = [{
            "complex_id": "1TST:LIG:A:1", "pdb_id": "1tst", "resname": "LIG", "chain": "A",
            "resseq": "1", "uniprot": "P0", "fda_status": "4.0", "arm": arm, "k": k,
            "mask_seed": 42 + k, "gen_seed": 1000 + k,
            "parent_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "masked_smiles": masked, "n_atoms": 13, "pool_size": 5,
            "n_bpe_tokens": 13, "budget_tokens": 1, "n_sampled_atoms": 1,
            "n_mask_tokens": 1, "mask_token_frac": 0.0769,
            "masked_atom_indices": "[0]", "pool_source": "selftest", "skip_reason": "",
        } for arm in bc.ARMS for k in range(2)]
        import build_masks as bm
        bc.write_csv_rows(P["masks"], mask_rows, bm.MASK_FIELDS)
        digest = bc.sha256_file(P["masks"])
        bc.write_manifest(td, "masks", {"masks_csv_sha256": digest})

        assert verify_masks_unchanged(td, P["masks"], strict=True) == digest
        with open(P["masks"], "a", encoding="utf-8") as f:
            f.write("\n")
        try:
            verify_masks_unchanged(td, P["masks"], strict=True)
            raise AssertionError("expected SystemExit on mask drift")
        except SystemExit:
            print("  ✓ mask-drift guard: edited masks.csv stops the run (sha256 mismatch)")

        pred = [{"complex_id": "1TST:LIG:A:1", "arm": "random", "k": 0,
                 "generated_smiles": "CCO", "valid": 1.0}]
        bc.write_csv_rows(P["predictions"], pred, PRED_FIELDS)
        done = {(r["complex_id"], r["arm"], str(r["k"]))
                for r in bc.read_csv_rows(P["predictions"])}
        todo = [r for r in mask_rows if (r["complex_id"], r["arm"], str(r["k"])) not in done]
        assert len(todo) == len(mask_rows) - 1, (len(todo), len(mask_rows))
        print(f"  ✓ resume: {len(mask_rows)} mask rows, 1 already done -> {len(todo)} to generate")

    print("\nALL SELF-TESTS PASSED")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill the frozen masks with vanilla ChemBERTa.")
    p.add_argument("--out-dir", default=None, help="Bundle root (default config.BASELINE_DIR).")
    p.add_argument("--model", default=None,
                   help="HF checkpoint to fill masks with (default config.CHEMBERTA_MODEL). "
                        "Point this at another model to re-run the IDENTICAL masks.")
    p.add_argument("--top-k", type=int, default=None, help="Override BASELINE_TOP_K.")
    p.add_argument("--temperature", type=float, default=None, help="Override BASELINE_TEMPERATURE.")
    p.add_argument("--overwrite", action="store_true", help="Ignore an existing predictions.csv.")
    p.add_argument("--allow-mask-drift", action="store_true",
                   help="Proceed even if masks.csv no longer matches the manifest hash.")
    p.add_argument("--flush-every", type=int, default=25, help="Checkpoint predictions.csv every N rows.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N mask rows (debugging).")
    p.add_argument("--test", action="store_true", help="Run the self-test and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.test:
        return _self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

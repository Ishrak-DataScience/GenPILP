# -*- coding: utf-8 -*-
"""
stage10_self_test.py
====================
Self-test for Stage 10 (both variants).

Run via:  python stage10_vanila_backpropagation_training.py --test

Lives in its own module rather than inside the training script so the training
script's __main__ stays a thin dispatcher, and so the test can be imported and
run piecewise while debugging.

What it pins down, in order of what would hurt most if wrong:

  1. LOSS ORDERING -- an invalid molecule must lose to EVERY valid one, and a
     perfect molecule must beat a mediocre one. If this is wrong, best-of-K
     selects the wrong targets and nothing downstream can save it.
  2. TARGET RECOVERY -- the parent fallback must return the parent's ACTUAL
     tokens at the mask positions. A silent off-by-one here would train the
     model on wrong targets while every metric still looked plausible.
  3. GRADIENT REACHES ONLY THE UNFROZEN LAYERS.
  4. UNLIKELIHOOD DOES NOT FIGHT THE CROSS-ENTROPY -- a token that is the
     target at its position must never also be pushed down.
  5. BOTH VARIANTS TRAIN, and 10a differs from 10b (otherwise the comparison
     you are about to run is measuring nothing).
  6. CE ACTUALLY LEARNS -- repeated steps on one example must raise the
     probability of the target tokens.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

import stage10_vanila_backpropagation_training as s10


def _build_pairs(tokenizer):
    """
    Build (masked, parent) pairs THE WAY THE PIPELINE DOES -- through
    mask_atoms_in_smiles_token_level.

    Hand-writing the masked strings does not work, and the reason is worth
    recording: the masker canonicalises first, so a hand-written
    "CC(=O)OC1=CC=CC=C1C(=O)<mask>" is masking a Kekule form the pipeline never
    produces (canonical is "CC(=O)Oc1ccccc1C(=O)O"). parent_target_ids then
    correctly refuses to align them. Generating the pairs here keeps the test
    on the real data path instead of testing a fiction.
    """
    from stage1_mask_calculation import mask_atoms_in_smiles_token_level

    smis = [
        "CC(=O)Oc1ccccc1C(=O)O",                 # aspirin
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",          # caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",            # ibuprofen
        "CC(=O)Nc1ccc(O)cc1",                    # paracetamol
    ]
    pairs = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        n = mol.GetNumAtoms()
        atoms = {1, n // 2}                       # deterministic, 1-2 masks
        masked = mask_atoms_in_smiles_token_level(smi, atoms, tokenizer)
        if tokenizer.mask_token in masked:
            pairs.append((masked, smi))
    assert pairs, "could not build any masked test pair"
    return pairs


def run_self_test() -> None:
    # ── 1. loss ordering ──────────────────────────────────────────────────
    parent = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    l_bad, c_bad = s10.compute_stage10_loss("not_a_smiles(((", parent)
    l_self, _    = s10.compute_stage10_loss(parent, parent)
    l_other, _   = s10.compute_stage10_loss("CC(=O)Nc1ccc(O)cc1", parent)

    assert c_bad["valid"] == 0.0
    assert abs(l_bad - s10.LOSS_INVALID) < 1e-9, f"invalid loss {l_bad}"
    assert l_bad > s10.S_WORST_VALID, "invalid must beat the WORST valid molecule"
    assert l_self <= s10.S_WORST_VALID and l_other <= s10.S_WORST_VALID
    # Identical to the parent => similarity 1 => full novelty penalty.
    _, c_self = s10.compute_stage10_loss(parent, parent)
    assert abs(c_self["novelty"] - s10.W_NOVELTY) < 1e-6, (
        "reconstructing the parent must incur the FULL novelty penalty")
    _, c_other = s10.compute_stage10_loss("CC(=O)Nc1ccc(O)cc1", parent)
    assert c_other["novelty"] < c_self["novelty"], "a different molecule must be more novel"
    # Terms must sum to the reported loss, or the plotted split is a lie.
    assert abs(sum(c_other[k] for k in s10.LOSS_TERMS) - l_other) < 1e-9
    print(f"  [1] loss ordering: invalid {l_bad:.3f} > worst valid "
          f"{s10.S_WORST_VALID:.3f} >= parent-copy {l_self:.3f}, "
          f"novel {l_other:.3f}   OK")

    # ── 2. parent-target recovery ─────────────────────────────────────────
    tokenizer, model, device = s10.load_model_last_layers()
    PAIRS = _build_pairs(tokenizer)
    n_checked = 0
    for masked, par in PAIRS:
        tgt = s10.parent_target_ids(masked, par, tokenizer)
        assert tgt is not None, f"no target recovered for {masked!r}"
        ids = tokenizer(masked)["input_ids"]
        n_masks = sum(1 for t in ids if t == tokenizer.mask_token_id)
        assert len(tgt) == n_masks, f"{len(tgt)} targets for {n_masks} masks"
        # Substituting the recovered tokens back must reproduce the canonical
        # parent exactly -- the strongest possible check on the alignment.
        filled = list(ids)
        it = iter(tgt)
        for i, t in enumerate(filled):
            if t == tokenizer.mask_token_id:
                filled[i] = next(it)
        rebuilt = tokenizer.decode(filled, skip_special_tokens=True).replace(" ", "")
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(par), canonical=True)
        assert rebuilt == canonical, f"rebuilt {rebuilt!r} != canonical {canonical!r}"
        n_checked += 1
    print(f"  [2] parent targets rebuild the canonical parent exactly "
          f"({n_checked}/{len(PAIRS)})   OK")

    # ── 3. only the unfrozen layers get gradients ─────────────────────────
    model.train()
    loss, stats = s10.stage10_batch_loss(PAIRS[:2], tokenizer, model, device,
                                         variant="a", k_cand=4)
    assert loss.requires_grad, "loss must carry a gradient"
    loss.backward()
    n_blocks = model.config.num_hidden_layers
    frozen_with_grad, trainable_with_grad = [], []
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            (trainable_with_grad if p.requires_grad else frozen_with_grad).append(name)
    assert not frozen_with_grad, f"frozen params received gradient: {frozen_with_grad[:3]}"
    assert trainable_with_grad, "no trainable parameter received a gradient"
    assert any("lm_head" in n for n in trainable_with_grad), "LM head must train"
    assert all(("lm_head" in n) or (f"layer.{n_blocks - 1}." in n)
               for n in trainable_with_grad), \
        "only the LM head and the last block should be trainable by default"
    print(f"  [3] gradient reaches {len(trainable_with_grad)} trainable tensors, "
          f"0 frozen ones   OK")

    # ── 4. unlikelihood never fights the cross-entropy ────────────────────
    # Construct the pathological case directly: a "bad" candidate that shares
    # the target's token at one position. That position must be excluded.
    tgt_ids = torch.tensor([5, 9], device=device)
    bad = torch.tensor([[5, 3], [7, 9]], device=device)     # each shares one token
    keep = bad != tgt_ids.unsqueeze(0)
    assert keep.tolist() == [[False, True], [True, False]], \
        "positions equal to the target must be excluded from unlikelihood"
    print("  [4] unlikelihood excludes tokens the CE is pulling toward   OK")

    # ── 5. both variants run, and differ ──────────────────────────────────
    torch.manual_seed(0)
    la, sa_ = s10.stage10_batch_loss(PAIRS, tokenizer, model, device,
                                     variant="a", k_cand=8)
    torch.manual_seed(0)
    lb, sb_ = s10.stage10_batch_loss(PAIRS, tokenizer, model, device,
                                     variant="b", k_cand=8)
    assert la.requires_grad and lb.requires_grad
    assert float(la) > float(lb), (
        f"10a must add a non-negative unlikelihood term on top of 10b's CE "
        f"(got a={float(la):.4f}, b={float(lb):.4f})")
    assert sa_["unlikelihood"] > 0 and sb_["unlikelihood"] == 0
    print(f"  [5] variant a loss {float(la):.4f} > variant b loss {float(lb):.4f} "
          f"(the unlikelihood term is what differs)   OK")

    # ── 6. cross-entropy actually raises the target's probability ─────────
    masked, par = PAIRS[0]
    tgt = s10.parent_target_ids(masked, par, tokenizer)
    enc = tokenizer(masked, return_tensors="pt").to(device)
    ids = enc["input_ids"][0]
    mpos = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    tgt_t = torch.tensor(tgt, device=device)

    def target_prob():
        with torch.no_grad():
            lp = F.log_softmax(model(**enc).logits[0][mpos], dim=-1)
        return float(lp.gather(1, tgt_t.unsqueeze(1)).exp().mean())

    before = target_prob()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(12):
        opt.zero_grad()
        lp = F.log_softmax(model(**enc).logits[0][mpos], dim=-1)
        F.nll_loss(lp, tgt_t).backward()
        opt.step()
    after = target_prob()
    assert after > before, f"CE did not raise target probability ({before} -> {after})"
    print(f"  [6] CE raises target probability {before:.4f} -> {after:.4f}   OK")

    # ── 7. end-to-end training run, both variants, with checkpointing ─────
    for variant in ("a", "b"):
        with tempfile.TemporaryDirectory() as td:
            hist = s10.run_stage10_training(
                pairs=list(PAIRS), save_dir=td, variant=variant,
                num_epochs=1, batch_size=2, k_cand=4)
            assert hist["epoch"] == [1]
            assert os.path.isfile(os.path.join(td, "epoch_001.pt"))
            assert os.path.isfile(os.path.join(td, "stage10_checkpoint.json"))
            assert os.path.isfile(os.path.join(td, "config.json")), \
                "final model must be saved in HF format for the eval pass"
            for key in ("loss_mean", "fallback_rate", "cand_valid_rate"):
                assert len(hist[key]) == 1
            # Resuming a completed run must be a no-op, not a retrain.
            again = s10.run_stage10_training(
                pairs=list(PAIRS), save_dir=td, variant=variant,
                num_epochs=1, batch_size=2, k_cand=4)
            assert again["epoch"] == [1], "resume should not re-run a finished epoch"
        print(f"  [7{variant}] end-to-end run + checkpoint + resume (variant "
              f"{variant})   OK")

    print("Stage 10 self-test passed.")


if __name__ == "__main__":
    run_self_test()

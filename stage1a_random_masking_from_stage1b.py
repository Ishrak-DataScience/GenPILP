#!/usr/bin/env python3
"""
stage1a_random_masking_from_stage1b.py
──────────────────────────────────────
    Stage 1b's PLIP summary CSV  →  unique valid parent SMILES  →  RANDOM
    token-level masking at config.STAGE1A_FROM_STAGE1B_MASK_PERCENT  →  a
    Stage-1a-format CSV that Stage 9a reads as its "random masking" panel.

WHY THIS EXISTS
    Stage 9a plots two panels side by side: "Random masking (Stage 1a)" and
    "PLIP masking (Stage 1b)". Ordinary stage1a_random_masking.py builds its
    panel from ChEMBL molecules, so the two panels differ in the MOLECULE SET
    and in the MASK CHOICE at the same time -- any gap between them could be
    either. This script removes that confound: it masks the SAME molecules
    Stage 1b masked, at the SAME percent, and only the choice of which tokens
    get masked differs (uniformly at random here, PLIP interaction pool
    there). That is the control arm the comparison actually needs.

NO GENERATION HAPPENS HERE
    Stage 9a generates its own completions from masked_smiles, so this script
    only has to produce (masked_smiles, smiles) pairs -- no ChemBERTa forward
    pass, no temperature. The `generated_smiles` / `valid` columns are written
    empty to keep the Stage 1a schema intact, and the filename's temp field is
    0 to signal "no generation performed".

DESIGN DECISIONS
D1. PARENTS: Stage 1b summary rows with status == "ok" and a non-empty
    `smiles` (Stage 1b writes the UNMASKED parent there, alongside its own
    masked_smiles / masked_atom_indices, which this script ignores).
D2. UNIQUE: deduplicated on CANONICAL SMILES, keeping the first occurrence in
    file order. Stage 1b writes one row per ligand INSTANCE
    (pdb_id:resname:chain:resseq), so a ligand solved in N PDB entries appears
    N times; without this the control arm would be instance-weighted while
    "unique valid SMILES" is what was asked for. Matches the dedup Stage 9a
    now applies to Stage 1b (config.STAGE9_EVAL_DEDUP_BY_PARENT).
D3. VALID: parents RDKit refuses are dropped -- PDB-derived SMILES include
    hypervalent P/S and unkekulizable rings, and QED / novelty against an
    unparseable parent are undefined anyway. Counted, never raised.
D4. MASK COUNT: config.STAGE1A_FROM_STAGE1B_ROUNDING picks floor(P% x tokens)
    (Stage 1b's rule, the fair head-to-head) or round() (ordinary Stage 1a's
    rule). Always at least one token, capped at the token count.
D5. SEEDING: the per-molecule mask seed is
    sha256(base_seed:canonical_smiles:percent) -- keyed on the MOLECULE, not
    its row index, so the mask a molecule gets is identical whether it was row
    5 or row 500,000, and re-running after Stage 1b grows reproduces every
    earlier mask. Reuses stage1a_random_masking.mask_seed.
D6. LENGTH: pairs whose masked form exceeds ChemBERTa's 512-token window are
    dropped here rather than being written and silently dropped later by
    Stage 9a's collector. Masking can LENGTHEN a string in tokens (each <mask>
    is its own token), so the masked form is what gets checked.
D7. OUTPUT: config.STAGE1A_FROM_STAGE1B_DIR /
    mask{percent}pct_temp0_seed{seed}.csv -- exactly the
    mask{STAGE9_MASK_PERCENT}pct_temp*_seed*.csv glob
    collect_pairs_from_stage1a looks for, with the Stage 1a column set plus
    pdb_id/resname/chain/resseq provenance columns (extra columns are ignored
    by the reader).

USAGE
    python stage1a_random_masking_from_stage1b.py
    python stage1a_random_masking_from_stage1b.py --percent 20 --seed 7
    python stage1a_random_masking_from_stage1b.py --limit 5000
    python stage1a_random_masking_from_stage1b.py --stage1b path/to/output.tar.gz
    python stage1a_random_masking_from_stage1b.py --test   (synthetic smoke test)

THEN point Stage 9a at it:
    config.STAGE9_9A_STAGE1A_DATA_DIR = config.STAGE1A_FROM_STAGE1B_DIR
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import config
from bpe_mask_adapter import apply_bpe_mask_spans
from stage1a_random_masking import CSV_FIELDS, mask_seed
from stage9_masked_property_finetune import (
    MAX_MODEL_TOKENS,
    _canonical_parent,
    _read_csv_rows_matching,
    get_chemberta_tokenizer,
)

STAGE1B_SUMMARY_CSV = "stage1b_large_scale_plip_mask_summary.csv"

# Stage 1a's columns first (so the file IS a Stage 1a file), then provenance.
OUT_FIELDS = CSV_FIELDS + ["pdb_id", "resname", "chain", "resseq"]


# ════════════════════════════════════════════════════════════════════════════
#  PARENTS  (D1, D2, D3)
# ════════════════════════════════════════════════════════════════════════════

def load_unique_valid_parents(
    stage1b_location: str,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Unique, RDKit-valid parent molecules from Stage 1b's summary CSV.

    Returns (parents, stats) where each parent is
    {"smiles", "canonical", "pdb_id", "resname", "chain", "resseq"} -- the
    provenance being that of the FIRST instance the molecule appeared as.
    `stage1b_location` may be a directory or a .tar/.tar.gz archive.
    """
    rows = _read_csv_rows_matching(stage1b_location, STAGE1B_SUMMARY_CSV)
    stats = {"rows": len(rows), "parents": 0, "status_not_ok": 0,
             "missing_smiles": 0, "invalid_parent": 0, "duplicate_parents": 0}
    if not rows:
        return [], stats

    seen: set = set()
    parents: List[Dict[str, str]] = []
    for row in rows:
        if row.get("status") != "ok":
            stats["status_not_ok"] += 1
            continue
        smi = (row.get("smiles") or "").strip()
        if not smi:
            stats["missing_smiles"] += 1
            continue
        if Chem.MolFromSmiles(smi) is None:          # D3
            stats["invalid_parent"] += 1
            continue
        canonical = _canonical_parent(smi)           # D2
        if canonical in seen:
            stats["duplicate_parents"] += 1
            continue
        seen.add(canonical)
        parents.append({
            "smiles":    smi,
            "canonical": canonical,
            "pdb_id":    row.get("pdb_id", ""),
            "resname":   row.get("resname", ""),
            "chain":     row.get("chain", ""),
            "resseq":    row.get("resseq", ""),
        })
        if limit and len(parents) >= limit:
            break

    stats["parents"] = len(parents)
    return parents, stats


# ════════════════════════════════════════════════════════════════════════════
#  RANDOM TOKEN MASKING  (D4, D5)
# ════════════════════════════════════════════════════════════════════════════

def mask_tokens_at_random(
    smiles:    str,
    percent:   float,
    tokenizer,
    seed:      int,
    rounding:  str = "floor",
) -> Tuple[str, int, int]:
    """
    Mask `percent`% of the SMILES' BPE tokens, chosen uniformly at random.

    Mirrors stage1a_random_masking.mask_smiles_tokens (same tokenizer, same
    offset-span replacement via the shared apply_bpe_mask_spans primitive) and
    adds `rounding` so the count rule can match Stage 1b's floor instead of
    Stage 1a's round -- see D4. Returns (masked_smiles, n_tokens, n_masked),
    or (smiles, 0, 0) when the SMILES yields no tokens.
    """
    import random

    enc = tokenizer(smiles, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"] if int(e) > int(s)]
    n_tokens = len(offsets)
    if n_tokens == 0:
        return smiles, 0, 0

    raw = percent / 100.0 * n_tokens
    n_mask = int(raw) if rounding == "floor" else round(raw)
    n_mask = min(max(1, n_mask), n_tokens)

    rng = random.Random(seed)
    spans = [offsets[i] for i in sorted(rng.sample(range(n_tokens), n_mask))]
    return apply_bpe_mask_spans(smiles, spans, tokenizer.mask_token), n_tokens, n_mask


def build_masked_rows(
    parents:   List[Dict[str, str]],
    percent:   float,
    base_seed: int,
    rounding:  str,
    tokenizer,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """
    One Stage-1a-format row per parent. Rows whose masked form carries no
    <mask> or busts ChemBERTa's window are dropped and counted (D6), so what
    lands in the CSV is exactly what Stage 9a will be able to use.
    """
    out:   List[Dict[str, object]] = []
    stats = {"masked": 0, "no_mask": 0, "over_length": 0, "mask_failed": 0}

    for p in parents:
        smi = p["smiles"]
        try:
            masked, n_tokens, n_masked = mask_tokens_at_random(
                smi, percent, tokenizer,
                mask_seed(base_seed, p["canonical"], percent),   # D5
                rounding,
            )
        except Exception:
            stats["mask_failed"] += 1
            continue

        if not masked or tokenizer.mask_token not in masked:
            stats["no_mask"] += 1
            continue
        if len(tokenizer(masked, add_special_tokens=True)["input_ids"]) > MAX_MODEL_TOKENS:
            stats["over_length"] += 1                            # D6
            continue

        out.append({
            "chembl_id":       (f"{p['pdb_id']}:{p['resname']}:"
                                f"{p['chain']}:{p['resseq']}"),
            "smiles":          smi,
            "mask_percent":    percent,
            "temperature":     0,
            "seed":            base_seed,
            "n_tokens":        n_tokens,
            "n_masked":        n_masked,
            "masked_smiles":   masked,
            "generated_smiles": "",     # Stage 9a generates its own
            "valid":           "",
            "pdb_id":          p["pdb_id"],
            "resname":         p["resname"],
            "chain":           p["chain"],
            "resseq":          p["resseq"],
        })

    stats["masked"] = len(out)
    return out, stats


def write_rows(rows: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def combo_csv_name(percent: float, seed: int) -> str:
    """Filename collect_pairs_from_stage1a's mask{P}pct_temp*_seed*.csv finds."""
    return f"mask{percent:g}pct_temp0_seed{seed}.csv"


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(
    stage1b_location: str   = None,
    output_dir:       str   = None,
    percent:          float = None,
    base_seed:        int   = None,
    rounding:         str   = None,
    limit:            int   = None,
) -> str:
    """Build the random-masking control arm. Returns the CSV path written."""
    stage1b_location = (stage1b_location
                        or getattr(config, "STAGE9_9A_STAGE1B_DATA_DIR", "")
                        or config.STAGE1B_PLIP_MASK_DIR)
    output_dir = output_dir or config.STAGE1A_FROM_STAGE1B_DIR
    percent    = config.STAGE1A_FROM_STAGE1B_MASK_PERCENT if percent   is None else percent
    base_seed  = config.STAGE1A_FROM_STAGE1B_SEED         if base_seed is None else base_seed
    rounding   = config.STAGE1A_FROM_STAGE1B_ROUNDING     if rounding  is None else rounding
    if limit is None:
        limit = config.STAGE1A_FROM_STAGE1B_LIMIT

    print("\n" + "=" * 60)
    print("STAGE 1a (from Stage 1b) -- RANDOM-MASKING CONTROL ARM")
    print("=" * 60)
    print(f"""
  Reads   : {stage1b_location}
            ({STAGE1B_SUMMARY_CSV}, `smiles` column = the UNMASKED parent)
  Masks   : {percent}% of each molecule's ChemBERTa BPE tokens, chosen
            uniformly at random ({rounding} rounding, min 1 token),
            per-molecule seed derived from base seed {base_seed}
  Molecules: every UNIQUE, RDKit-valid parent{f' (capped at {limit})' if limit else ''}
  Writes  : {os.path.join(output_dir, combo_csv_name(percent, base_seed))}
  No generation happens here -- Stage 9a fills the masks itself.
""")

    if percent != config.STAGE9_MASK_PERCENT:
        print(f"  WARNING: percent={percent} != config.STAGE9_MASK_PERCENT="
              f"{config.STAGE9_MASK_PERCENT}. Stage 9a globs for "
              f"mask{config.STAGE9_MASK_PERCENT:g}pct_temp*_seed*.csv and will NOT "
              f"find this file.\n")

    parents, pstats = load_unique_valid_parents(stage1b_location, limit)
    print(f"  {pstats['rows']} Stage 1b row(s) read -> {pstats['parents']} unique valid "
          f"parent(s)")
    print(f"    dropped: status_not_ok {pstats['status_not_ok']}, missing_smiles "
          f"{pstats['missing_smiles']}, invalid_parent {pstats['invalid_parent']}, "
          f"duplicate_parents {pstats['duplicate_parents']}")
    if not parents:
        print(f"\n  No usable parents found in {stage1b_location} -- nothing written. "
              f"Point --stage1b (or config.STAGE9_9A_STAGE1B_DATA_DIR / "
              f"config.STAGE1B_PLIP_MASK_DIR) at the directory or .tar.gz holding "
              f"{STAGE1B_SUMMARY_CSV}.")
        sys.exit(1)

    tokenizer = get_chemberta_tokenizer()
    rows, mstats = build_masked_rows(parents, percent, base_seed, rounding, tokenizer)
    print(f"  {mstats['masked']} molecule(s) masked")
    print(f"    dropped: no_mask {mstats['no_mask']}, over_length "
          f"{mstats['over_length']}, mask_failed {mstats['mask_failed']}")
    if not rows:
        print("\n  Every parent was dropped during masking -- nothing written.")
        sys.exit(1)

    out_path = os.path.join(output_dir, combo_csv_name(percent, base_seed))
    write_rows(rows, out_path)

    avg = sum(int(r["n_masked"]) for r in rows) / len(rows)
    print(f"\n  Masked tokens per molecule: mean {avg:.2f}")
    print(f"  Written : {out_path}")
    print(f"""
  To make Stage 9a plot this as its "Random masking (Stage 1a)" panel, set

      STAGE9_9A_STAGE1A_DATA_DIR = "{output_dir}"

  in config.py, then re-run stage9a_masked_property_without_finetuning.py.
  Both panels will then cover the SAME molecules at the SAME {percent}% --
  only random vs. PLIP mask choice differs.""")
    print("=" * 60)
    return out_path


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    import tempfile

    tokenizer = get_chemberta_tokenizer()

    # ── D4: percent honoured, both rounding rules, min 1 token ──────────────
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    n_all = len(tokenizer(smi, add_special_tokens=False)["input_ids"])
    for rule in ("floor", "round"):
        masked, n_tokens, n_masked = mask_tokens_at_random(smi, 15, tokenizer, 1, rule)
        assert n_tokens == n_all, (n_tokens, n_all)
        expected = int(0.15 * n_all) if rule == "floor" else round(0.15 * n_all)
        assert n_masked == min(max(1, expected), n_all), (rule, n_masked, expected)
        assert masked.count(tokenizer.mask_token) == n_masked, masked
    assert mask_tokens_at_random("C", 1, tokenizer, 1)[2] == 1, "min one token"
    assert mask_tokens_at_random(smi, 100, tokenizer, 1)[2] == n_all, "capped at n_tokens"

    # ── D5: same molecule -> same mask regardless of position/run ───────────
    canon = _canonical_parent(smi)
    a = mask_tokens_at_random(smi, 15, tokenizer, mask_seed(42, canon, 15))
    b = mask_tokens_at_random(smi, 15, tokenizer, mask_seed(42, canon, 15))
    assert a == b, "same seed must reproduce the same mask"
    # Different base seeds must not all collapse to one mask. Checked over a
    # spread rather than a single pair: at floor(15%) only one token of ~13 is
    # masked, so any two seeds coincide often enough to make a pairwise
    # assertion flaky.
    spread = {mask_tokens_at_random(smi, 15, tokenizer, mask_seed(s, canon, 15))[0]
              for s in range(20)}
    assert len(spread) > 1, "base seed should change which token is masked"

    # ── D1/D2/D3: parent loading from a synthetic Stage 1b CSV ──────────────
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, STAGE1B_SUMMARY_CSV)
        with open(src, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "pdb_id", "resname", "chain", "resseq", "masking_mode", "status",
                "error", "smiles", "masked_smiles", "bpe_mask_count",
                "masked_atom_indices"])
            w.writeheader()
            rows = [
                ("1abc", "LIG", "A", "1", "ok", "c1ccccc1"),       # benzene
                ("2def", "LIG", "B", "2", "ok", "C1=CC=CC=C1"),    # same, Kekule -> dup
                ("3ghi", "LIG", "A", "3", "ok", "c1ccccc1"),       # exact dup
                ("4jkl", "LIG", "A", "4", "ok", "CC(=O)Oc1ccccc1C(=O)O"),
                ("5mno", "LIG", "A", "5", "ok", "not_a_smiles"),   # invalid
                ("6pqr", "LIG", "A", "6", "ok", ""),               # missing
                ("7stu", "LIG", "A", "7", "failed", "CCO"),        # status
            ]
            for pdb, res, ch, seq, status, s in rows:
                w.writerow({"pdb_id": pdb, "resname": res, "chain": ch, "resseq": seq,
                            "masking_mode": "attractive", "status": status, "error": "",
                            "smiles": s, "masked_smiles": "", "bpe_mask_count": "0",
                            "masked_atom_indices": "[]"})

        parents, pstats = load_unique_valid_parents(tmp)
        assert pstats["rows"] == 7, pstats
        assert pstats["status_not_ok"] == 1 and pstats["missing_smiles"] == 1, pstats
        assert pstats["invalid_parent"] == 1, pstats
        assert pstats["duplicate_parents"] == 2, pstats      # Kekule + exact
        assert [p["smiles"] for p in parents] == ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
        assert parents[0]["pdb_id"] == "1abc", "keeps FIRST instance's provenance"

        assert len(load_unique_valid_parents(tmp, limit=1)[0]) == 1, "limit"

        # ── D7: written file is readable by Stage 9a's own collector ────────
        out_dir = os.path.join(tmp, "out")
        path = main(stage1b_location=tmp, output_dir=out_dir, percent=15,
                    base_seed=42, rounding="floor")
        assert os.path.basename(path) == "mask15pct_temp0_seed42.csv", path

        from stage9_masked_property_finetune import collect_pairs_from_stage1a
        pairs = collect_pairs_from_stage1a(out_dir, percent=15)
        assert len(pairs) == 2, pairs
        for masked, orig in pairs:
            assert tokenizer.mask_token in masked and Chem.MolFromSmiles(orig)
        assert {o for _, o in pairs} == {"c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"}

        # rerunning reproduces the file byte for byte (D5)
        again = main(stage1b_location=tmp, output_dir=os.path.join(tmp, "out2"),
                     percent=15, base_seed=42, rounding="floor")
        assert open(path, encoding="utf-8").read() == open(again, encoding="utf-8").read()

    print("\nStage 1a-from-1b self-test passed.")


def _parse_args(argv: List[str]) -> dict:
    out: dict = {}
    flags = {"--stage1b": "stage1b_location", "--out": "output_dir",
             "--percent": "percent", "--seed": "base_seed",
             "--rounding": "rounding", "--limit": "limit"}
    for flag, key in flags.items():
        if flag not in argv:
            continue
        i = argv.index(flag)
        if i + 1 >= len(argv):
            print(f"  {flag} needs a value.")
            sys.exit(2)
        raw = argv[i + 1]
        if key == "percent":
            out[key] = float(raw)
        elif key == "limit":
            out[key] = None if raw.lower() in ("none", "all", "0") else int(raw)
        elif key == "base_seed":
            out[key] = int(raw)          # 0 is a perfectly good seed
        else:
            out[key] = raw
    if out.get("rounding") not in (None, "floor", "round"):
        print("  --rounding must be 'floor' or 'round'.")
        sys.exit(2)
    return out


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main(**_parse_args(sys.argv[1:]))

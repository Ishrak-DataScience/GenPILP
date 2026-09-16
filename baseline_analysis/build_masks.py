# -*- coding: utf-8 -*-
"""
build_masks.py
==============
Step 2 of the baseline analysis: freeze the INPUT DATA for every arm.

For each complex in selection.csv this writes, per complex:

    complexes/<PDB>_<LIG>_<CHAIN>_<POS>/
        complex.pdb      decompressed crystal structure (docking input)
        plip.xml         the PLIP report it was masked from
        pools.json       the three atom pools + provenance for this complex

and, once for the whole bundle:

    masks.csv            one row per (complex, arm, mask variant k) -- the
                         masked SMILES the model will be shown, the exact atom
                         indices drawn, and the realised mask-token count
    mask_pools.json      every complex's full pools, for re-deriving masks
    manifest.json        ["masks"] section: settings + sha256 of masks.csv

WHY THIS STEP IS SEPARATE FROM GENERATION
------------------------------------------
"The masks should be saved as well in order to reproduce eventually with a
different model (we must guarantee that in this case the data is exactly the
same)." So the masks are a frozen, hashed artifact produced BEFORE any model is
loaded. generate_predictions.py consumes masks.csv and never re-derives a mask;
pointing it at a different checkpoint re-runs the identical inputs, and the
manifest's sha256 proves it.

THE THREE ARMS (see baseline_common for the full argument)
-----------------------------------------------------------
    plip_pos  PLIP ++  atoms PLIP reports as contacting the protein
    plip_neg  PLIP --  atoms PLIP reports as NOT contacting it
    random             every atom, uniform

All three are drawn to the same budget: floor(BASELINE_MASK_PERCENT% of the
parent's BPE tokens), via stage9_masked_property_finetune._remask_from_pool --
the same function Stage 9 trains with. A pool smaller than the budget is masked
whole, which is why PLIP++ masks <= the other two arms.

POOL SOURCE, AND THE COMPLEMENT SHORTCUT
-----------------------------------------
Default: read the pre-computed Stage 1b PLIP-- corpus
(config.STAGE1B_PLIP_NEGATIVE_MASK_DIR), whose masked_atom_indices IS the
plip_neg pool; plip_pos is its complement over the parent's atom indices, which
is exactly how Stage 1 builds mode 2 (masked = all - attractive).

--pools-from plip re-runs Stage 1's own run_pipeline on the staged PDB + XML
instead (mode 2 for plip_neg, mode 1 for plip_pos) -- slower, needs the PLIP
mirror, and is the fallback for complexes the corpus lacks.

--verify-plip-pos runs BOTH and asserts they agree, so the complement shortcut
is checked rather than trusted. Mismatches are reported per complex and, unless
--keep-going, abort the run.

HOW TO RUN
-----------
    # HPC, the normal path (corpus + PDB/XML mirrors from config)
    python baseline_analysis/build_masks.py --out-dir "$BASELINE_DIR"

    # same, but also prove the PLIP++ complement against a real Stage 1b mode-1 run
    python baseline_analysis/build_masks.py --verify-plip-pos

    # masks only, no PDB staging (no PDB mirror needed -- docking comes later)
    python baseline_analysis/build_masks.py --no-stage-pdb

    # self-test (synthetic corpus + the checked-in 100d fixture if present)
    python baseline_analysis/build_masks.py --test
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):     # repo root (config, stageN) + this dir
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_common as bc  # noqa: E402

MASK_FIELDS = [
    "complex_id", "pdb_id", "resname", "chain", "resseq", "uniprot", "fda_status",
    "arm", "k", "mask_seed", "gen_seed",
    "parent_smiles", "masked_smiles",
    "n_atoms", "pool_size", "n_bpe_tokens", "budget_tokens",
    "n_sampled_atoms", "n_mask_tokens", "mask_token_frac",
    "masked_atom_indices", "pool_source", "skip_reason",
]

t0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


# ════════════════════════════════════════════════════════════════════════════
#  STAGING THE STRUCTURE FILES
# ════════════════════════════════════════════════════════════════════════════

def _resolve_pdb(pdb_root: str, pdb_id: str) -> Optional[str]:
    """
    Find one structure in the local mirror. The Stage 1b layout
    (<root>/<mid2>/pdb<id>.ent.gz) is tried first; the other spellings let the
    same script run against a flat directory of PDB files.
    """
    pid = pdb_id.lower()
    mid2 = pid[1:3]
    for rel in (
        os.path.join(mid2, f"pdb{pid}.ent.gz"),
        os.path.join(mid2, f"pdb{pid}.ent"),
        f"pdb{pid}.ent.gz",
        f"{pid}.pdb.gz",
        f"{pid}.pdb",
        f"{pid.upper()}.pdb",
        f"{pid}.ent",
        pid.upper(),
        pid,
    ):
        cand = os.path.join(pdb_root, rel)
        if os.path.isfile(cand):
            return cand
    return None


def _resolve_xml(xml_root: str, pdb_id: str) -> Optional[str]:
    pid = pdb_id.lower()
    for rel in (f"pdb{pid}.xml", f"{pid}.xml", f"{pid.upper()}.xml",
                f"{pid.upper()}_report.xml", f"pdb{pid}_report.xml"):
        cand = os.path.join(xml_root, rel)
        if os.path.isfile(cand):
            return cand
    return None


def stage_structure(
    root: str, complex_id: str, pdb_root: str, xml_root: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Copy this complex's PDB (decompressed) and PLIP XML into its bundle
    directory. Returns (pdb_path, xml_path, problems) -- either path is None
    when the mirror does not have it, which is a WARNING here: masks do not
    need the structure, only docking does.
    """
    pdb_id, resname, chain, resseq = bc.parse_complex_id(complex_id)
    dest = bc.complex_dir(root, complex_id)
    os.makedirs(dest, exist_ok=True)
    problems: List[str] = []

    out_pdb = os.path.join(dest, "complex.pdb")
    src_pdb = _resolve_pdb(pdb_root, pdb_id)
    if src_pdb is None:
        problems.append(f"no PDB for {pdb_id} under {pdb_root}")
        out_pdb = None
    elif not (os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0):
        opener = gzip.open if src_pdb.endswith(".gz") else open
        with opener(src_pdb, "rb") as fin, open(out_pdb, "wb") as fout:
            shutil.copyfileobj(fin, fout)

    out_xml = os.path.join(dest, "plip.xml")
    src_xml = _resolve_xml(xml_root, pdb_id)
    if src_xml is None:
        problems.append(f"no PLIP XML for {pdb_id} under {xml_root}")
        out_xml = None
    elif not (os.path.exists(out_xml) and os.path.getsize(out_xml) > 0):
        shutil.copyfile(src_xml, out_xml)

    return out_pdb, out_xml, problems


# ════════════════════════════════════════════════════════════════════════════
#  POOLS FROM A LIVE STAGE 1 RUN  (fallback / verification path)
# ════════════════════════════════════════════════════════════════════════════

def pools_from_stage1(
    pdb_path: str, xml_path: str, resname: str, chain: str, resseq: str,
    mask_non_attractive: bool,
) -> Tuple[str, List[int]]:
    """
    Run Stage 1's own masking for one binding site and return
    (parent_smiles, pool). mask_non_attractive=True gives the PLIP-- pool,
    False the PLIP++ pool -- identical code to Stage 1b's two modes.
    """
    import stage1_mask_calculation as s1

    # config.INCLUDE_TYPES, exactly as Stage 1b passes it: the PLIP-- corpus was
    # built with that set, so the live fallback has to use the same one or the
    # two pool sources would disagree about what "interacting" means.
    include = list(getattr(bc.config, "INCLUDE_TYPES", []) or
                   ["hydrophobic", "hbond", "waterBridge", "saltBridge",
                    "piStacking", "piCation", "halogen", "metal"])
    with tempfile.TemporaryDirectory() as td:
        meta = s1.run_pipeline(
            pdb_path=pdb_path, plip_xml_path=xml_path,
            resname=resname, chain=(chain or None),
            resseq=(int(resseq) if str(resseq).strip().lstrip("-").isdigit() else None),
            include_types=include, representation="selfies", mask_token="<mask>",
            out_prefix=os.path.join(td, "x"), serial_map_json=None,
            mask_non_attractive=mask_non_attractive,
            save_meta=False, save_plot=False,
        )
    return meta["smiles"], [int(i) for i in meta["masked_atom_indices"]]


# ════════════════════════════════════════════════════════════════════════════
#  MASK BUILDING
# ════════════════════════════════════════════════════════════════════════════

def masks_for_complex(
    row: dict,
    parent_smiles: str,
    pools: Dict[str, List[int]],
    tokenizer,
    percent: float,
    n_seeds: int,
    mask_seed: int,
    gen_seed: int,
    pool_source: str,
) -> List[dict]:
    """
    Every mask row for one complex: n_seeds variants per arm.

    The k-th variant uses mask_seed + k for the atom draw and gen_seed + k for
    the later MLM sampling. When an arm's pool is <= the budget the draw is the
    whole pool, so its k variants share one masked string by construction and
    differ only in the model's sampling seed -- that is the honest behaviour for
    PLIP++, and analyze.py reports distinct-mask counts so it stays visible.
    """
    complex_id = row["id"]
    pdb_id, resname, chain, resseq = bc.parse_complex_id(complex_id)
    n_atoms = bc.n_heavy_atoms(parent_smiles)
    n_tokens, budget = bc.mask_budget(parent_smiles, tokenizer, percent)

    out: List[dict] = []
    for arm in bc.ARMS:
        pool = pools[arm]
        for k in range(n_seeds):
            ms, gs = mask_seed + k, gen_seed + k
            seed_key = f"{complex_id}|{arm}"
            masked, reason, sampled = bc.build_mask(
                parent_smiles, pool, percent, tokenizer, seed_key, ms,
            )
            n_mask_tokens = bc.count_mask_tokens(masked, tokenizer) if masked else 0
            out.append({
                "complex_id": complex_id, "pdb_id": pdb_id, "resname": resname,
                "chain": chain, "resseq": resseq,
                "uniprot": row.get("UniProt ID", ""), "fda_status": row.get("fda_status", ""),
                "arm": arm, "k": k, "mask_seed": ms, "gen_seed": gs,
                "parent_smiles": parent_smiles, "masked_smiles": masked,
                "n_atoms": n_atoms, "pool_size": len(pool),
                "n_bpe_tokens": n_tokens, "budget_tokens": budget,
                "n_sampled_atoms": len(sampled), "n_mask_tokens": n_mask_tokens,
                "mask_token_frac": round(n_mask_tokens / n_tokens, 4) if n_tokens else "",
                "masked_atom_indices": json.dumps(sampled),
                "pool_source": pool_source, "skip_reason": reason or "",
            })
    return out


def assert_budget_respected(rows: Sequence[dict], percent: float) -> None:
    """
    The one invariant the whole comparison rests on: no arm may mask more than
    `percent`% of a parent's BPE tokens. Checked on the realised rows rather
    than trusted from the sampling code.
    """
    bad = []
    for r in rows:
        if r["skip_reason"]:
            continue
        n_tok, n_mask = int(r["n_bpe_tokens"]), int(r["n_mask_tokens"])
        if n_mask > int(r["budget_tokens"]) or (n_tok and n_mask / n_tok > percent / 100.0 + 1e-9):
            bad.append(f"{r['complex_id']}/{r['arm']}/k{r['k']}: "
                       f"{n_mask} masks of {n_tok} tokens (budget {r['budget_tokens']})")
        if int(r["n_sampled_atoms"]) > int(r["pool_size"]):
            bad.append(f"{r['complex_id']}/{r['arm']}/k{r['k']}: drew more atoms than the pool holds")
    if bad:
        raise AssertionError("mask budget violated:\n  " + "\n  ".join(bad))


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> int:
    root = bc.bundle_dir(args.out_dir)
    P = bc.paths(root)
    os.makedirs(root, exist_ok=True)

    sel_path = args.selection or P["selection"]
    if not os.path.exists(sel_path):
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "selection.csv")
        if os.path.exists(local):
            sel_path = local
        else:
            print(f"ERROR: no selection.csv at {P['selection']} (or {local}).\n"
                  f"Run  python baseline_analysis/select_samples.py  first.")
            return 2
    selection = bc.read_csv_rows(sel_path)
    if args.limit:
        selection = selection[: args.limit]
    log(f"selection: {len(selection)} complexes from {sel_path}")

    percent   = float(args.mask_percent if args.mask_percent is not None
                      else bc.baseline_settings()["mask_percent"])
    n_seeds   = int(args.seeds if args.seeds is not None else bc.baseline_settings()["n_seeds"])
    mask_seed = int(bc.baseline_settings()["mask_seed"])
    gen_seed  = int(bc.baseline_settings()["gen_seed"])

    pdb_root = args.pdb_root or getattr(bc.config, "PLIP_LARGE_SCALE_PDB_ROOT", "")
    xml_root = args.xml_root or getattr(bc.config, "PLIP_LARGE_SCALE_XML_ROOT", "")
    corpus   = args.corpus or getattr(bc.config, "STAGE1B_PLIP_NEGATIVE_MASK_DIR", "")

    # ── stage structures ────────────────────────────────────────────────────
    staged: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    if args.no_stage_pdb:
        log("skipping PDB/XML staging (--no-stage-pdb)")
    else:
        problems: List[str] = []
        for row in selection:
            pdb_p, xml_p, probs = stage_structure(root, row["id"], pdb_root, xml_root)
            staged[row["id"]] = (pdb_p, xml_p)
            problems.extend(probs)
        n_ok = sum(1 for v in staged.values() if v[0])
        log(f"staged structures: {n_ok}/{len(selection)} PDB, "
            f"{sum(1 for v in staged.values() if v[1])}/{len(selection)} XML")
        for p in problems[:10]:
            print(f"  ⚠️  {p}")
        if problems and len(problems) > 10:
            print(f"  ⚠️  ... and {len(problems) - 10} more")

    # ── pools ───────────────────────────────────────────────────────────────
    keys = [bc.parse_complex_id(r["id"]) for r in selection]
    corpus_rows: Dict[Tuple[str, str, str, str], dict] = {}
    if args.pools_from in ("corpus", "auto"):
        if not corpus:
            log("no PLIP-- corpus configured (STAGE1B_PLIP_NEGATIVE_MASK_DIR)")
        else:
            log(f"reading PLIP-- corpus: {corpus}")
            corpus_rows = bc.read_negative_rows(corpus, keys)
            log(f"corpus rows matched: {len(corpus_rows)}/{len(keys)}")

    tokenizer = None
    import stage9_masked_property_finetune as s9
    tokenizer = s9.get_chemberta_tokenizer()

    all_rows: List[dict] = []
    pools_dump: Dict[str, dict] = {}
    verify_fail: List[str] = []
    skipped: List[str] = []

    for row in selection:
        cid = row["id"]
        key = bc.parse_complex_id(cid)
        pdb_p, xml_p = staged.get(cid, (None, None))
        parent_smiles, neg_pool, source = "", [], ""

        crow = corpus_rows.get(key)
        if crow is not None and args.pools_from in ("corpus", "auto"):
            parent_smiles = (crow.get("smiles") or "").strip()
            try:
                neg_pool = [int(i) for i in json.loads(crow.get("masked_atom_indices") or "[]")]
                source = "stage1b_corpus"
            except (json.JSONDecodeError, TypeError, ValueError):
                neg_pool, source = [], ""

        if (not parent_smiles or not neg_pool) and args.pools_from in ("plip", "auto"):
            if pdb_p and xml_p:
                try:
                    parent_smiles, neg_pool = pools_from_stage1(
                        pdb_p, xml_p, key[1], key[2], key[3], mask_non_attractive=True)
                    source = "stage1_live"
                except Exception as e:
                    skipped.append(f"{cid}: live PLIP-- masking failed ({e})")
                    continue
            else:
                skipped.append(f"{cid}: no corpus row and no staged PDB/XML to mask from")
                continue

        if not parent_smiles or not neg_pool:
            skipped.append(f"{cid}: no PLIP-- pool available")
            continue

        try:
            pools = bc.pools_for_parent(parent_smiles, neg_pool)
        except ValueError as e:
            skipped.append(f"{cid}: {e}")
            continue

        # Prove the complement shortcut against a real Stage 1b mode-1 run.
        verified = None
        if args.verify_plip_pos:
            if not (pdb_p and xml_p):
                verify_fail.append(f"{cid}: cannot verify, no staged PDB/XML")
            else:
                try:
                    smi1, pos_live = pools_from_stage1(
                        pdb_p, xml_p, key[1], key[2], key[3], mask_non_attractive=False)
                    same_smiles = (smi1 == parent_smiles)
                    same_pool = sorted(pos_live) == sorted(pools["plip_pos"])
                    verified = bool(same_smiles and same_pool)
                    if not verified:
                        verify_fail.append(
                            f"{cid}: complement != live PLIP++ "
                            f"(smiles_match={same_smiles}, "
                            f"complement={len(pools['plip_pos'])} atoms, live={len(pos_live)})")
                except Exception as e:
                    verify_fail.append(f"{cid}: live PLIP++ masking failed ({e})")

        rows = masks_for_complex(row, parent_smiles, pools, tokenizer, percent,
                                 n_seeds, mask_seed, gen_seed, source)
        all_rows.extend(rows)

        pools_dump[cid] = {
            "parent_smiles": parent_smiles,
            "n_atoms": bc.n_heavy_atoms(parent_smiles),
            "pools": {a: pools[a] for a in bc.ARMS},
            "pool_source": source,
            "plip_pos_verified": verified,
            "pdb": os.path.relpath(pdb_p, root) if pdb_p else None,
            "plip_xml": os.path.relpath(xml_p, root) if xml_p else None,
            "uniprot": row.get("UniProt ID", ""),
            "fda_status": row.get("fda_status", ""),
        }
        cdir = bc.complex_dir(root, cid)
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, "pools.json"), "w", encoding="utf-8") as f:
            json.dump(pools_dump[cid], f, indent=2)

    log(f"built masks for {len(pools_dump)} complexes -> {len(all_rows)} rows")
    for s in skipped:
        print(f"  ⚠️  skipped {s}")
    if verify_fail:
        print("\n  PLIP++ verification problems:")
        for v in verify_fail:
            print(f"    ✗ {v}")
        if not args.keep_going:
            print("\nERROR: --verify-plip-pos found mismatches; not writing masks.csv.\n"
                  "       Re-run with --keep-going to write anyway.")
            return 3
    elif args.verify_plip_pos:
        log("PLIP++ complement verified against live Stage 1 mode-1 masking for every complex ✓")

    if not all_rows:
        print("ERROR: no masks built.")
        return 4

    assert_budget_respected(all_rows, percent)
    log(f"budget invariant holds: no arm exceeds {percent}% of BPE tokens ✓")

    bc.write_csv_rows(P["masks"], all_rows, MASK_FIELDS)
    with open(P["mask_pools"], "w", encoding="utf-8") as f:
        json.dump(pools_dump, f, indent=2)

    usable = [r for r in all_rows if not r["skip_reason"]]
    by_arm = {a: [r for r in usable if r["arm"] == a] for a in bc.ARMS}
    distinct = {a: len({(r["complex_id"], r["masked_smiles"]) for r in by_arm[a]}) for a in bc.ARMS}
    bc.write_manifest(root, "masks", {
        "settings": {**bc.baseline_settings(), "mask_percent": percent, "n_seeds": n_seeds},
        "selection_csv": os.path.abspath(sel_path),
        "corpus": corpus,
        "pdb_root": pdb_root, "xml_root": xml_root,
        "pools_from": args.pools_from,
        "plip_pos_verified": bool(args.verify_plip_pos and not verify_fail),
        "n_complexes": len(pools_dump), "n_mask_rows": len(all_rows),
        "n_usable_rows": len(usable),
        "distinct_masks_per_arm": distinct,
        "skipped": skipped,
        "masks_csv_sha256": bc.sha256_file(P["masks"]),
        "mask_pools_sha256": bc.sha256_file(P["mask_pools"]),
        "tokenizer": bc.baseline_settings()["chemberta_model"],
    })

    print("\n── mask summary ──────────────────────────────────────────────")
    print(f"{'arm':<10}{'rows':>6}{'usable':>8}{'distinct':>10}"
          f"{'mean pool':>11}{'mean masks':>12}{'mean %tok':>11}")
    for a in bc.ARMS:
        rs = by_arm[a]
        if not rs:
            print(f"{a:<10}{0:>6}{0:>8}{0:>10}{'-':>11}{'-':>12}{'-':>11}")
            continue
        mp = sum(int(r['pool_size']) for r in rs) / len(rs)
        mm = sum(int(r['n_mask_tokens']) for r in rs) / len(rs)
        mf = sum(float(r['mask_token_frac'] or 0) for r in rs) / len(rs) * 100
        print(f"{a:<10}{len([r for r in all_rows if r['arm']==a]):>6}{len(rs):>8}"
              f"{distinct[a]:>10}{mp:>11.1f}{mm:>12.2f}{mf:>10.1f}%")
    reasons: Dict[str, int] = {}
    for r in all_rows:
        if r["skip_reason"]:
            reasons[r["skip_reason"]] = reasons.get(r["skip_reason"], 0) + 1
    if reasons:
        print("\nunusable rows by reason: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    print(f"\nmasks.csv      : {P['masks']}")
    print(f"mask_pools.json: {P['mask_pools']}")
    print(f"manifest.json  : {P['manifest']}  (sha256 of masks.csv recorded)")
    return 0


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ════════════════════════════════════════════════════════════════════════════

def _self_test() -> int:
    import csv as _csv

    print("build_masks.py self-test")
    import stage9_masked_property_finetune as s9
    tok = s9.get_chemberta_tokenizer()

    # Aspirin-ish parent with a known atom count; pool 0..3 are "non-interacting".
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    n_atoms = bc.n_heavy_atoms(smiles)
    neg = [0, 1, 2, 3]
    pools = bc.pools_for_parent(smiles, neg)
    assert pools["plip_neg"] == neg, pools["plip_neg"]
    assert pools["plip_pos"] == [i for i in range(n_atoms) if i not in neg]
    assert pools["random"] == list(range(n_atoms))
    assert set(pools["plip_pos"]) & set(pools["plip_neg"]) == set()
    assert len(pools["plip_pos"]) + len(pools["plip_neg"]) == n_atoms
    print(f"  ✓ pools: {n_atoms} atoms -> pos={len(pools['plip_pos'])} "
          f"neg={len(pools['plip_neg'])} random={n_atoms}; complement exact, disjoint")

    # Out-of-range pool must be refused, not silently intersected.
    try:
        bc.pools_for_parent(smiles, [0, 999])
        raise AssertionError("expected ValueError for out-of-range pool")
    except ValueError:
        print("  ✓ out-of-range PLIP-- pool rejected")

    # Budget + determinism.
    n_tok, budget = bc.mask_budget(smiles, tok, 15)
    row = {"id": "1TST:ASP:A:1", "UniProt ID": "P00000", "fda_status": "4.0"}
    rows = masks_for_complex(row, smiles, pools, tok, 15, 3, 42, 1000, "selftest")
    assert len(rows) == 3 * len(bc.ARMS)
    assert_budget_respected(rows, 15)
    print(f"  ✓ budget: {n_tok} BPE tokens -> floor(15%) = {budget} masks max; "
          f"{len(rows)} rows within budget")

    rows2 = masks_for_complex(row, smiles, pools, tok, 15, 3, 42, 1000, "selftest")
    assert [r["masked_smiles"] for r in rows] == [r["masked_smiles"] for r in rows2]
    assert [r["masked_atom_indices"] for r in rows] == [r["masked_atom_indices"] for r in rows2]
    print("  ✓ determinism: identical masks and atom draws on a re-run (same seeds)")

    for r in rows:
        if r["skip_reason"]:
            continue
        drawn = json.loads(r["masked_atom_indices"])
        assert set(drawn) <= set(pools[r["arm"]]), (r["arm"], drawn)
        assert len(drawn) <= int(r["pool_size"])
        assert int(r["n_mask_tokens"]) <= int(r["budget_tokens"]), (r["n_mask_tokens"], r)
    print("  ✓ every drawn atom comes from its own arm's pool; realised <mask> "
          "tokens <= budget")

    # The invariant that Stage 9's atom-count rule does NOT hold: a molecule made
    # of multi-token bracket atoms must still come in under the TOKEN budget.
    multi = "C[S@](=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1"  # lapatinib
    n_tok_m, budget_m = bc.mask_budget(multi, tok, 15)
    pools_m = bc.pools_for_parent(multi, list(range(bc.n_heavy_atoms(multi) // 2)))
    worst = 0
    for arm in bc.ARMS:
        for k in range(5):
            ms, reason, drawn = bc.build_mask(multi, pools_m[arm], 15, tok,
                                              f"lapatinib|{arm}", 42 + k)
            if reason:
                continue
            got = bc.count_mask_tokens(ms, tok)
            worst = max(worst, got)
            assert got <= budget_m, (arm, k, got, budget_m)
    print(f"  ✓ token budget on a bracket-atom-heavy drug (lapatinib): "
          f"{n_tok_m} tokens, budget {budget_m}, worst realised mask {worst} "
          f"(Stage 9's atom-count rule overshoots here)")

    small = bc.pools_for_parent(smiles, list(range(n_atoms)))
    tiny = masks_for_complex(row, smiles, {"plip_pos": [5], "plip_neg": small["plip_neg"],
                                           "random": small["random"]},
                             tok, 15, 3, 42, 1000, "selftest")
    pos = [r for r in tiny if r["arm"] == "plip_pos" and not r["skip_reason"]]
    assert pos and len({r["masked_smiles"] for r in pos}) == 1
    assert all(int(r["n_sampled_atoms"]) == 1 for r in pos)
    print("  ✓ pool smaller than budget: whole pool masked, one distinct mask across seeds")

    # A parent RDKit cannot parse is reported, never raised.
    bad = bc.build_mask("this-is-not-smiles", [0], 15, tok, "k", 42)
    assert bad[1] == "invalid_parent", bad
    print("  ✓ unparseable parent -> skip_reason='invalid_parent', no exception")

    # End-to-end against the real PLIP-- corpus, if it is on this machine.
    corpus = getattr(bc.config, "STAGE1B_PLIP_NEGATIVE_MASK_DIR", "")
    sel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "selection.csv")
    if corpus and os.path.exists(sel):
        picks = bc.read_csv_rows(sel)[:3]
        got = bc.read_negative_rows(corpus, [bc.parse_complex_id(r["id"]) for r in picks])
        if got:
            for pr in picks:
                k = bc.parse_complex_id(pr["id"])
                if k not in got:
                    continue
                cr = got[k]
                pl = bc.pools_for_parent(cr["smiles"],
                                         json.loads(cr["masked_atom_indices"]))
                rs = masks_for_complex(pr, cr["smiles"], pl, tok, 15, 2, 42, 1000, "corpus")
                assert_budget_respected(rs, 15)
            print(f"  ✓ real corpus: pools + masks built for {len(got)}/{len(picks)} "
                  f"sampled complexes, budget holds")
        else:
            print("  (real corpus present but none of the sampled complexes matched)")
    else:
        print("  (skipped real-corpus check: no corpus or selection.csv on this machine)")

    # Optional: a local PDB+XML pair exercises the live Stage 1 path. The
    # configured mirrors are tried first, then the Dummy_data tree this repo is
    # usually checked out next to (which holds the 100d fixture on the laptop).
    def _fixture(pdb_id: str) -> Tuple[Optional[str], Optional[str]]:
        guesses = [
            (getattr(bc.config, "PLIP_LARGE_SCALE_PDB_ROOT", ""),
             getattr(bc.config, "PLIP_LARGE_SCALE_XML_ROOT", "")),
        ]
        up = os.path.dirname(bc.REPO_ROOT)
        for d in (os.path.join(up, "Dummy_data"), os.path.join(bc.REPO_ROOT, "Dataset")):
            guesses.append((os.path.join(d, "pdb"), d))
            guesses.append((d, d))
        for pr, xr in guesses:
            if not pr or not xr:
                continue
            pp, xx = _resolve_pdb(pr, pdb_id), _resolve_xml(xr, pdb_id)
            if pp and xx:
                return pp, xx
        return None, None

    fx_pdb, fx_xml = _fixture("100d")
    if fx_pdb and fx_xml:
        print(f"  (live Stage 1 fixture found: {os.path.basename(fx_pdb)} + "
              f"{os.path.basename(fx_xml)}; checking mode1/mode2 agreement)")
        import stage1b_large_scale_PLIP_mask_calculation as s1b
        checked = 0
        _fx_tmp = tempfile.mkdtemp()
        if fx_pdb.endswith(".gz"):      # Stage 1 reads PDB as text; decompress first
            plain = os.path.join(_fx_tmp, "fixture.pdb")
            with gzip.open(fx_pdb, "rb") as fin, open(plain, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            fx_pdb = plain
        for resname, chain, resseq in s1b.list_binding_sites(fx_xml):
            try:
                smi_n, neg_p = pools_from_stage1(fx_pdb, fx_xml, resname, chain, str(resseq), True)
                smi_p, pos_p = pools_from_stage1(fx_pdb, fx_xml, resname, chain, str(resseq), False)
            except Exception as e:
                print(f"    ({resname}:{chain}:{resseq} skipped: {str(e)[:70]})")
                continue
            if smi_n != smi_p:
                print(f"    ({resname}:{chain}:{resseq} skipped: mode1/mode2 SMILES differ)")
                continue
            comp = bc.pools_for_parent(smi_n, neg_p)["plip_pos"]
            assert sorted(comp) == sorted(pos_p), (resname, comp, pos_p)
            checked += 1
        if checked:
            print(f"  ✓ live fixture: PLIP++ == complement(PLIP--) on {checked} real binding site(s)")
        else:
            print("  (fixture has no PLIP interactions to mask, so the complement "
                  "identity cannot be exercised here; run --verify-plip-pos on HPC)")
    else:
        print("  (no live PDB+XML fixture on this machine; covered by --verify-plip-pos on HPC)")

    with tempfile.TemporaryDirectory() as td:
        bc.write_csv_rows(os.path.join(td, "masks.csv"), rows, MASK_FIELDS)
        with open(os.path.join(td, "masks.csv"), newline="", encoding="utf-8") as f:
            back = list(_csv.DictReader(f))
        assert len(back) == len(rows) and set(back[0]) == set(MASK_FIELDS)
        h1 = bc.sha256_file(os.path.join(td, "masks.csv"))
        bc.write_csv_rows(os.path.join(td, "masks2.csv"), rows2, MASK_FIELDS)
        assert h1 == bc.sha256_file(os.path.join(td, "masks2.csv"))
    print("  ✓ masks.csv round-trips and is byte-identical across runs (reproducibility hash)")

    print("\nALL SELF-TESTS PASSED")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the three masking arms for the baseline analysis.")
    p.add_argument("--out-dir", default=None, help="Bundle root (default config.BASELINE_DIR).")
    p.add_argument("--selection", default=None, help="selection.csv to use (default <out-dir>/selection.csv).")
    p.add_argument("--corpus", default=None,
                   help="Stage 1b PLIP-- summary CSV, its directory, or a tar "
                        "(default config.STAGE1B_PLIP_NEGATIVE_MASK_DIR).")
    p.add_argument("--pdb-root", default=None, help="Local PDB mirror (default config.PLIP_LARGE_SCALE_PDB_ROOT).")
    p.add_argument("--xml-root", default=None, help="PLIP XML dir (default config.PLIP_LARGE_SCALE_XML_ROOT).")
    p.add_argument("--pools-from", choices=["auto", "corpus", "plip"], default="auto",
                   help="auto (default): corpus, falling back to a live Stage 1 run per complex.")
    p.add_argument("--verify-plip-pos", action="store_true",
                   help="Also run Stage 1 mode 1 and assert PLIP++ == complement(PLIP--).")
    p.add_argument("--keep-going", action="store_true",
                   help="Write masks.csv even if verification found mismatches.")
    p.add_argument("--no-stage-pdb", action="store_true", help="Do not copy PDB/XML into the bundle.")
    p.add_argument("--mask-percent", type=float, default=None, help="Override BASELINE_MASK_PERCENT.")
    p.add_argument("--seeds", type=int, default=None, help="Override BASELINE_N_SEEDS.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N complexes (debugging).")
    p.add_argument("--test", action="store_true", help="Run the self-test and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.test:
        return _self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

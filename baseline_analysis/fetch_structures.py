# -*- coding: utf-8 -*-
"""
fetch_structures.py
===================
Put a `complex.pdb` next to every selected complex, so docking can run on a
machine that has no PDB mirror.

Order of preference, per complex:

  1. already in the bundle            -> left alone (nothing is re-downloaded)
  2. the local PDB mirror             -> copied/decompressed
     (config.PLIP_LARGE_SCALE_PDB_ROOT, Stage 1b layout)
  3. RCSB                             -> https://files.rcsb.org/download/<id>.pdb

WHY THIS EXISTS
----------------
build_masks.py stages structures from the mirror, but the mirror is a group
filesystem that is not mounted everywhere. Masks do NOT need it (the PLIP--
corpus row carries the parent SMILES and the atom pools); only docking does, and
only for the handful of complexes actually selected. 24 structures is a ~25 MB
download, so an HPC account with no mirror can still run the whole baseline.

The PLIP XML is NOT fetched: nothing downstream needs it once masks.csv exists.
Re-running build_masks.py --verify-plip-pos does, and that needs the mirror.

HOW TO RUN
-----------
    python baseline_analysis/fetch_structures.py --out-dir "$BASELINE_DIR"
    python baseline_analysis/fetch_structures.py --out-dir "$BASELINE_DIR" --no-download
    python baseline_analysis/fetch_structures.py --test
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import time
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_common as bc  # noqa: E402

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

t0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def looks_like_pdb(path: str, resname: str = "", chain: str = "", resseq: str = "") -> bool:
    """
    A usable structure has ATOM records and, when asked, the actual ligand --
    docking needs the HETATM block to place the autobox, so a structure missing
    it is worse than useless (it would dock into an arbitrary box).
    """
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return False
    has_atom = False
    has_lig = not resname
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("ATOM"):
                has_atom = True
            elif resname and line.startswith("HETATM"):
                if (line[17:20].strip() == resname
                        and (not chain or line[21].strip() == chain)
                        and (not resseq or line[22:26].strip() == str(resseq))):
                    has_lig = True
            if has_atom and has_lig:
                return True
    return has_atom and has_lig


def from_mirror(pdb_root: str, pdb_id: str, dest: str) -> bool:
    import build_masks as bm

    src = bm._resolve_pdb(pdb_root, pdb_id) if pdb_root else None
    if not src:
        return False
    opener = gzip.open if src.endswith(".gz") else open
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with opener(src, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return True


def from_rcsb(pdb_id: str, dest: str, timeout: int = 60) -> bool:
    """Download one entry. Writes via a temp file so a failed download leaves nothing."""
    import urllib.request

    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if os.path.getsize(tmp) == 0:
            raise OSError("empty response")
        os.replace(tmp, dest)
        return True
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"    download failed for {pdb_id}: {type(e).__name__}: {e}")
        return False


def run(args: argparse.Namespace) -> int:
    root = bc.bundle_dir(args.out_dir)
    P = bc.paths(root)
    sel_path = args.selection or P["selection"]
    if not os.path.exists(sel_path):
        print(f"ERROR: no selection.csv at {sel_path}.")
        return 2
    rows = bc.read_csv_rows(sel_path)
    pdb_root = args.pdb_root or getattr(bc.config, "PLIP_LARGE_SCALE_PDB_ROOT", "")

    have = mirrored = downloaded = failed = 0
    problems: List[str] = []
    for r in rows:
        cid = r["id"]
        pdb_id, resname, chain, resseq = bc.parse_complex_id(cid)
        dest = os.path.join(bc.complex_dir(root, cid), "complex.pdb")

        if looks_like_pdb(dest, resname, chain, resseq):
            have += 1
            continue
        if from_mirror(pdb_root, pdb_id, dest) and looks_like_pdb(dest, resname, chain, resseq):
            mirrored += 1
            continue
        if args.no_download:
            failed += 1
            problems.append(f"{cid}: not in the bundle or the mirror, and --no-download was given")
            continue
        log(f"  downloading {pdb_id.upper()} for {cid}")
        if from_rcsb(pdb_id, dest, args.timeout) and looks_like_pdb(dest, resname, chain, resseq):
            downloaded += 1
        else:
            failed += 1
            problems.append(f"{cid}: no usable structure (missing ATOM records or "
                            f"no HETATM {resname}:{chain}:{resseq})")

    log(f"structures: {have} already present, {mirrored} from mirror, "
        f"{downloaded} downloaded, {failed} failed")
    for p in problems:
        print(f"  ⚠️  {p}")
    bc.write_manifest(root, "structures", {
        "n_complexes": len(rows), "present": have, "from_mirror": mirrored,
        "downloaded": downloaded, "failed": failed,
        "pdb_root": pdb_root, "source": RCSB_URL,
        "problems": problems,
    })
    if failed:
        print(f"\n{failed} complex(es) have no structure; run_docking.py will emit an "
              f"error row for each and carry on with the rest.")
    return 0


def _self_test() -> int:
    import tempfile

    print("fetch_structures.py self-test")
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "good.pdb")
        with open(good, "w") as f:
            f.write("ATOM      1  CA  ALA A   1      11.104   6.134  -6.504  1.00 20.00           C\n")
            f.write("HETATM  100  C1  LIG A 600      10.000   5.000  -5.000  1.00 20.00           C\n")
            f.write("END\n")
        assert looks_like_pdb(good)
        assert looks_like_pdb(good, "LIG", "A", "600")
        assert not looks_like_pdb(good, "XXX", "A", "600"), "wrong ligand must not pass"
        assert not looks_like_pdb(good, "LIG", "B", "600"), "wrong chain must not pass"
        print("  ✓ looks_like_pdb: needs ATOM records AND the requested ligand's HETATM block")

        only_het = os.path.join(td, "het.pdb")
        with open(only_het, "w") as f:
            f.write("HETATM  100  C1  LIG A 600      10.000   5.000  -5.000  1.00 20.00           C\n")
        assert not looks_like_pdb(only_het), "a receptor-less file must not pass"
        assert not looks_like_pdb(os.path.join(td, "nope.pdb"))
        print("  ✓ looks_like_pdb: no ATOM records / missing file -> False")

        # Mirror path, including the .ent.gz layout Stage 1b uses.
        mirror = os.path.join(td, "mirror", "bc")
        os.makedirs(mirror, exist_ok=True)
        with gzip.open(os.path.join(mirror, "pdb1bcd.ent.gz"), "wb") as f:
            f.write(open(good, "rb").read())
        dest = os.path.join(td, "out", "complex.pdb")
        assert from_mirror(os.path.join(td, "mirror"), "1bcd", dest)
        assert looks_like_pdb(dest, "LIG", "A", "600")
        print("  ✓ from_mirror: <root>/<mid2>/pdb<id>.ent.gz decompressed into the bundle")
        assert not from_mirror(os.path.join(td, "mirror"), "9zzz", dest + "2")
        print("  ✓ from_mirror: absent id -> False (caller falls through to RCSB)")

        # A failed download must leave no partial file behind.
        bad = os.path.join(td, "bad", "complex.pdb")
        ok = from_rcsb("ZZZZ", bad, timeout=5)
        assert not os.path.exists(bad + ".part")
        print(f"  ✓ from_rcsb: bad id -> {ok}, no .part file left behind")

        # End to end with --no-download: present + mirror, one genuine miss.
        P = bc.paths(td)
        sel = [{"id": "1BCD:LIG:A:600"}, {"id": "9ZZZ:LIG:A:1"}]
        bc.write_csv_rows(P["selection"], sel, ["id"])
        rc = run(argparse.Namespace(out_dir=td, selection=None,
                                    pdb_root=os.path.join(td, "mirror"),
                                    no_download=True, timeout=5))
        assert rc == 0
        assert looks_like_pdb(os.path.join(bc.complex_dir(td, "1BCD:LIG:A:600"), "complex.pdb"))
        assert not os.path.exists(os.path.join(bc.complex_dir(td, "9ZZZ:LIG:A:1"), "complex.pdb"))
        print("  ✓ end to end (--no-download): mirror hit staged, miss reported, exit 0")

    print("\nALL SELF-TESTS PASSED")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage complex.pdb for every selected complex.")
    p.add_argument("--out-dir", default=None, help="Bundle root (default config.BASELINE_DIR).")
    p.add_argument("--selection", default=None, help="selection.csv (default <bundle>/selection.csv).")
    p.add_argument("--pdb-root", default=None, help="Local mirror (default config.PLIP_LARGE_SCALE_PDB_ROOT).")
    p.add_argument("--no-download", action="store_true", help="Never contact RCSB.")
    p.add_argument("--timeout", type=int, default=60, help="Per-download timeout (s).")
    p.add_argument("--test", action="store_true", help="Run the self-test and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.test:
        return _self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

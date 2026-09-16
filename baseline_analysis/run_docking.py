# -*- coding: utf-8 -*-
"""
run_docking.py
==============
Step 4: dock the reference and every prediction into each complex's own pocket.

Per complex in the bundle this produces, under docking/<PDB>_<LIG>_<CH>_<POS>/:

    rec.pdb                    receptor (ATOM records)
    orig.pdb                   crystal ligand (HETATM) -- the autobox reference
    crystal.sdf                crystal ligand with bond orders from the parent SMILES
    parent_redock/             the crystal ligand re-docked from SMILES
    plip_pos/k0 ... random/k4  one directory per prediction
        ligand.sdf  docked_poses.sdf  docking.log

and one docking_summary.csv for the whole bundle: one row per (complex, arm, k,
pose) with GNINA's CNNscore, CNNaffinity and minimizedAffinity.

THE THREE REFERENCE ROWS, AND WHY THERE ARE THREE
--------------------------------------------------
"Are the predictions better than the redocked example" only means something
against a reference that got the SAME treatment, so the primary reference is
arm="parent_redock": the crystal ligand's own SMILES, embedded with ETKDG and
docked exactly like a prediction. Two extra rows guard that comparison:

  arm="parent_crystal"  GNINA --score_only on the crystal pose itself. The
                        experimental answer, no search involved -- it says
                        whether the docking setup is sane at all.
  redock_rmsd           heavy-atom RMSD between the redocked pose 1 and the
                        crystal pose, written into the summary for the
                        parent_redock row. The standard sanity threshold is
                        <= 2.0 A; above that, the pocket/box is suspect and
                        analyze.py flags that complex rather than trusting its
                        deltas.

GNINA IS LINUX-ONLY (Stage 6 A1). On any other platform this script prepares
every input and stops before docking, so the bundle can be built on a laptop and
the docking run on the cluster. Resumable: a molecule whose docked_poses.sdf
already parses is not re-docked (Stage 6 A11), so an interrupted or extended run
just gets relaunched.

HOW TO RUN
-----------
    python baseline_analysis/run_docking.py --out-dir "$BASELINE_DIR"
    python baseline_analysis/run_docking.py --prepare-only     # inputs, no GNINA
    python baseline_analysis/run_docking.py --complex 1ERR:RAL:B:600
    python baseline_analysis/run_docking.py --test
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_common as bc  # noqa: E402

DOCK_FIELDS = [
    "complex_id", "pdb_id", "resname", "chain", "resseq", "uniprot", "fda_status",
    "arm", "k", "smiles", "pose",
    "CNNscore", "CNNaffinity", "minimizedAffinity",
    "redock_rmsd", "status", "note", "sdf",
]

PARENT_REDOCK = bc.PARENT_ARM          # "parent_redock"
PARENT_CRYSTAL = "parent_crystal"

# Every SDF property name this run saw, so a GNINA that renames a score can be
# reported instead of silently yielding empty columns.
PROPS_SEEN: set = set()

t0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


# ════════════════════════════════════════════════════════════════════════════
#  GNINA
# ════════════════════════════════════════════════════════════════════════════

def gnina_cmd(
    gnina_bin: str, receptor: str, ligand: str, autobox: str,
    out_sdf: str, log_file: str, n_poses: int, exhaustiveness: int, seed: int,
    score_only: bool = False,
) -> List[str]:
    """
    The GNINA invocation. Beyond Stage 6's, this pins --seed and
    --exhaustiveness so a docking run is reproducible and its search effort is
    recorded, and supports --score_only for scoring the crystal pose in place.
    """
    cmd = [gnina_bin, "-r", receptor, "-l", ligand]
    if score_only:
        cmd += ["--score_only"]
    else:
        cmd += ["--autobox_ligand", autobox, "--out", out_sdf,
                "--num_modes", str(n_poses), "--exhaustiveness", str(exhaustiveness),
                "--seed", str(seed)]
    cmd += ["--log", log_file]
    return cmd


def default_runner(cmd: List[str], timeout: int = 3600) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout or "", p.stderr or ""


def gnina_version(gnina_bin: str) -> str:
    """
    `gnina --version`, recorded in the manifest.

    Provenance, not decoration: GNINA v1.3 rebuilt CNN scoring on Torch and
    RETRAINED the scoring functions on CrossDock2020 v1.3, so a CNNaffinity from
    v1.0.3 and one from v1.3.x are not the same quantity. The pipeline moved to
    v1.3.3 on 2026-09-17, so anything docked before that is superseded; recording
    the version is the only way to tell whether two runs can be compared -- or
    merged.
    """
    try:
        p = subprocess.run([gnina_bin, "--version"], capture_output=True, text=True, timeout=120)
        return " ".join((p.stdout or p.stderr or "").split())[:200] or "(no output)"
    except Exception as e:
        return f"(unavailable: {type(e).__name__})"


# Canonical score -> property spellings seen across GNINA versions. Matched
# case-insensitively; the first hit wins.
_SCORE_ALIASES = {
    "CNNscore":          ("CNNscore", "CNN_score", "cnnscore"),
    "CNNaffinity":       ("CNNaffinity", "CNN_affinity", "cnnaffinity"),
    "minimizedAffinity": ("minimizedAffinity", "minimized_affinity", "Affinity"),
}


def parse_gnina_scores(sdf_path: str) -> Tuple[List[Dict[str, str]], set]:
    """
    Per-pose scores from a GNINA output SDF, plus the set of property names the
    file actually carried.

    Deliberately not Stage 6's parse_gnina_scores, which reads three hard-coded
    property names: if a future GNINA renames one, that returns "" for every
    pose and the analysis quietly reports "no docking data" instead of failing.
    Here the aliases are explicit and the observed property names are handed back
    so run() can say exactly what it found.
    """
    from rdkit import Chem

    rows: List[Dict[str, str]] = []
    seen: set = set()
    if not os.path.exists(sdf_path):
        return rows, seen
    pose = 0
    for mol in Chem.ForwardSDMolSupplier(sdf_path, removeHs=False, sanitize=False):
        if mol is None:
            continue
        pose += 1
        props = mol.GetPropsAsDict()
        seen.update(props.keys())
        lower = {str(k).lower(): v for k, v in props.items()}
        row: Dict[str, str] = {"pose": pose}
        for canon, names in _SCORE_ALIASES.items():
            val = ""
            for n in names:
                if n.lower() in lower:
                    val = lower[n.lower()]
                    break
            row[canon] = val
        rows.append(row)
    return rows, seen


def parse_score_only(stdout: str, log_text: str = "") -> Dict[str, str]:
    """
    Pull the three scores out of a --score_only run, which prints them rather
    than writing an SDF. Missing values stay empty strings.
    """
    out: Dict[str, str] = {"CNNscore": "", "CNNaffinity": "", "minimizedAffinity": ""}
    keys = {"CNNscore": "CNNscore", "CNNaffinity": "CNNaffinity",
            "Affinity": "minimizedAffinity", "minimizedAffinity": "minimizedAffinity"}
    for line in (stdout + "\n" + log_text).splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[0] in keys and not out[keys[parts[0]]]:
            try:
                out[keys[parts[0]]] = str(float(parts[1]))
            except ValueError:
                pass
    return out


# ════════════════════════════════════════════════════════════════════════════
#  LIGAND PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def crystal_sdf_from_pdb(orig_pdb: str, parent_smiles: str, out_sdf: str) -> Optional[str]:
    """
    Crystal ligand as an SDF with correct bond orders, by mapping the parent
    SMILES onto the HETATM block. Returns the path, or None when the template
    will not fit (PDB ligands are missing hydrogens and sometimes atoms, so this
    is best-effort: everything downstream treats it as optional).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromPDBFile(orig_pdb, removeHs=True, sanitize=False)
    if mol is None:
        return None
    tmpl = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
    fixed = mol
    if tmpl is not None:
        try:
            fixed = AllChem.AssignBondOrdersFromTemplate(tmpl, mol)
            Chem.SanitizeMol(fixed)
        except Exception:
            fixed = mol
    try:
        Chem.SanitizeMol(fixed)
    except Exception:
        pass
    os.makedirs(os.path.dirname(out_sdf) or ".", exist_ok=True)
    w = Chem.SDWriter(out_sdf)
    w.write(fixed)
    w.close()
    return out_sdf if os.path.getsize(out_sdf) > 0 else None


def pose_rmsd(crystal_sdf: str, docked_sdf: str) -> Optional[float]:
    """
    Heavy-atom RMSD between the crystal ligand and docked pose 1 -- IN PLACE.

    rdMolAlign.CalcRMS, deliberately NOT GetBestRMS: GetBestRMS superimposes the
    two conformers before measuring, so a pose translated clean out of the
    pocket still scores ~0. What this number has to answer is "did the search put
    the ligand back where the crystal has it", which is the un-aligned distance.
    Symmetry-equivalent atom numberings are still resolved (CalcRMS enumerates
    substructure matches), so a flipped phenyl is not penalised.

    None when either molecule is unreadable or the graphs differ.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    try:
        ref = next(iter(Chem.SDMolSupplier(crystal_sdf, removeHs=True)), None)
        probe = next(iter(Chem.SDMolSupplier(docked_sdf, removeHs=True)), None)
        if ref is None or probe is None:
            return None
        ref, probe = Chem.RemoveHs(ref), Chem.RemoveHs(probe)
        if ref.GetNumAtoms() != probe.GetNumAtoms():
            return None
        return round(float(rdMolAlign.CalcRMS(probe, ref)), 3)
    except Exception:
        return None


def already_docked(sdf_path: str) -> bool:
    """Stage 6 A11: a molecule counts as docked when its SDF parses to >=1 pose."""
    from rdkit import Chem

    if not (os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0):
        return False
    try:
        return any(m is not None for m in
                   Chem.ForwardSDMolSupplier(sdf_path, removeHs=False, sanitize=False))
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════════════
#  ONE COMPLEX
# ════════════════════════════════════════════════════════════════════════════

def dock_complex(
    root: str,
    complex_id: str,
    meta: dict,
    ligands: List[Tuple[str, str, str]],     # (arm, k, smiles)
    gnina_bin: str,
    opts: argparse.Namespace,
    runner: Callable[[List[str]], Tuple[int, str, str]] = None,
) -> List[dict]:
    """Prepare inputs, dock every ligand, return docking_summary rows."""
    import stage6_1_docking as s6

    runner = runner or (lambda cmd: default_runner(cmd, timeout=opts.timeout))
    pdb_id, resname, chain, resseq = bc.parse_complex_id(complex_id)
    work = os.path.join(bc.paths(root)["docking_work"], complex_id.replace(":", "_"))
    os.makedirs(work, exist_ok=True)

    base = {"complex_id": complex_id, "pdb_id": pdb_id, "resname": resname,
            "chain": chain, "resseq": resseq,
            "uniprot": meta.get("uniprot", ""), "fda_status": meta.get("fda_status", "")}
    rows: List[dict] = []

    src_pdb = meta.get("pdb_path")
    if not (src_pdb and os.path.exists(src_pdb)):
        return [{**base, "arm": "", "k": "", "status": "error",
                 "note": "no complex.pdb staged (run build_masks.py with a PDB mirror)"}]

    rec = os.path.join(work, "rec.pdb")
    orig = os.path.join(work, "orig.pdb")
    if not os.path.exists(rec) and not s6.prepare_receptor(src_pdb, rec):
        return [{**base, "arm": "", "k": "", "status": "error", "note": "receptor prep failed"}]
    if not os.path.exists(orig):
        try:
            rs = int(resseq)
        except ValueError:
            rs = resseq
        if not s6.prepare_autobox_ligand(src_pdb, orig, resname, chain, rs):
            return [{**base, "arm": "", "k": "", "status": "error",
                     "note": f"no HETATM {resname}:{chain}:{resseq} in the PDB"}]

    parent_smiles = meta.get("parent_smiles", "")
    crystal = os.path.join(work, "crystal.sdf")
    if not os.path.exists(crystal):
        crystal_sdf_from_pdb(orig, parent_smiles, crystal)
    have_crystal = os.path.exists(crystal) and os.path.getsize(crystal) > 0

    if opts.prepare_only:
        return [{**base, "arm": "", "k": "", "status": "prepared",
                 "note": f"rec.pdb + orig.pdb{' + crystal.sdf' if have_crystal else ''}"}]

    # ── crystal pose, scored in place (no search) ───────────────────────────
    if have_crystal and not opts.no_crystal_score:
        lf = os.path.join(work, "crystal_score.log")
        rc, so, se = runner(gnina_cmd(gnina_bin, rec, crystal, orig, "", lf,
                                      opts.num_modes, opts.exhaustiveness, opts.seed,
                                      score_only=True))
        log_text = open(lf, encoding="utf-8", errors="replace").read() if os.path.exists(lf) else ""
        sc = parse_score_only(so, log_text)
        rows.append({**base, "arm": PARENT_CRYSTAL, "k": "", "smiles": parent_smiles,
                     "pose": 1, **sc, "redock_rmsd": 0.0,
                     "status": "ok" if rc == 0 else "error",
                     "note": "" if rc == 0 else se[-200:], "sdf": crystal})

    # ── redock + predictions ────────────────────────────────────────────────
    for arm, k, smiles in ligands:
        sub = os.path.join(work, arm if arm == PARENT_REDOCK else f"{arm}_k{k}")
        os.makedirs(sub, exist_ok=True)
        lig_sdf = os.path.join(sub, "ligand.sdf")
        out_sdf = os.path.join(sub, "docked_poses.sdf")
        log_f = os.path.join(sub, "docking.log")
        row0 = {**base, "arm": arm, "k": k, "smiles": smiles}

        if not already_docked(out_sdf):
            if not os.path.exists(lig_sdf) and not s6.smiles_to_sdf(smiles, lig_sdf, name=f"{complex_id}_{arm}_{k}"):
                rows.append({**row0, "status": "error", "note": "3D embedding failed"})
                continue
            rc, so, se = runner(gnina_cmd(gnina_bin, rec, lig_sdf, orig, out_sdf, log_f,
                                          opts.num_modes, opts.exhaustiveness, opts.seed))
            if rc != 0 or not already_docked(out_sdf):
                rows.append({**row0, "status": "error", "note": (se or so)[-200:]})
                continue

        rmsd = ""
        if arm == PARENT_REDOCK and have_crystal:
            r = pose_rmsd(crystal, out_sdf)
            rmsd = "" if r is None else r

        poses, props = parse_gnina_scores(out_sdf)
        PROPS_SEEN.update(props)
        if not poses:
            rows.append({**row0, "status": "error", "note": "no poses parsed", "sdf": out_sdf})
            continue
        for p in poses:
            rows.append({**row0, "pose": p["pose"],
                         "CNNscore": p.get("CNNscore", ""),
                         "CNNaffinity": p.get("CNNaffinity", ""),
                         "minimizedAffinity": p.get("minimizedAffinity", ""),
                         "redock_rmsd": rmsd if p["pose"] == 1 else "",
                         "status": "ok", "note": "", "sdf": out_sdf})
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def collect_jobs(root: str, only: Optional[List[str]]) -> Dict[str, dict]:
    """
    Group predictions.csv into per-complex docking jobs, each with its parent
    redock plus every valid, distinct prediction.

    Duplicate SMILES within an arm are docked ONCE and the score reused by
    analyze.py -- docking the same molecule twice costs GPU time and returns the
    same answer. An invalid prediction is not dockable and is left to the
    validity statistics.
    """
    import json

    P = bc.paths(root)
    preds = bc.read_csv_rows(P["predictions"])
    pools = {}
    if os.path.exists(P["mask_pools"]):
        with open(P["mask_pools"], encoding="utf-8") as f:
            pools = json.load(f)

    jobs: Dict[str, dict] = {}
    for r in preds:
        cid = r["complex_id"]
        if only and cid not in only:
            continue
        job = jobs.setdefault(cid, {
            "meta": {
                "uniprot": r.get("uniprot", ""), "fda_status": r.get("fda_status", ""),
                "parent_smiles": r.get("parent_smiles", ""),
                "pdb_path": os.path.join(bc.complex_dir(root, cid), "complex.pdb"),
            },
            "ligands": [], "seen": set(),
        })
        if str(r.get("valid")) not in ("1", "1.0"):
            continue
        smi = (r.get("generated_smiles") or "").strip()
        if not smi or smi in job["seen"]:
            continue
        job["seen"].add(smi)
        job["ligands"].append((r["arm"], r["k"], smi))

    for cid, job in jobs.items():
        parent = job["meta"]["parent_smiles"] or (pools.get(cid, {}) or {}).get("parent_smiles", "")
        job["meta"]["parent_smiles"] = parent
        if parent:
            job["ligands"].insert(0, (PARENT_REDOCK, "", parent))
        job["ligands"] = [(a, k, s) for a, k, s in job["ligands"]]
    return jobs


def shard_dir(root: str) -> str:
    return os.path.join(bc.paths(root)["docking_work"], "_shards")


def shard_path(root: str, shard: int) -> str:
    return os.path.join(shard_dir(root), f"docking_shard_{int(shard):04d}.csv")


def select_shard(complex_ids: List[str], shard: int, num_shards: int) -> List[str]:
    """
    Which complexes this array task owns: round-robin over the SORTED id list.

    Round-robin, not contiguous blocks, because complexes differ a lot in how
    many valid predictions they have -- interleaving spreads the big ones across
    tasks instead of loading them all onto one. Sorting first makes the split
    depend only on (shard, num_shards), so a re-run of one task redoes exactly
    the same complexes.
    """
    ids = sorted(complex_ids)
    return [c for i, c in enumerate(ids) if i % num_shards == shard]


def merge_shards(root: str) -> int:
    """
    Fold every shard CSV into docking_summary.csv, last write per key winning.

    Array tasks never touch docking_summary.csv -- 24 concurrent writers would
    interleave rows mid-file and corrupt it -- so each writes its own shard file
    and this runs once, after the array finishes.
    """
    P = bc.paths(root)
    merged: Dict[tuple, dict] = {}
    for r in bc.read_csv_rows(P["docking"]):
        merged[(r["complex_id"], r["arm"], str(r["k"]), str(r.get("pose", "")))] = r
    sd = shard_dir(root)
    files = sorted(glob.glob(os.path.join(sd, "docking_shard_*.csv"))) if os.path.isdir(sd) else []
    for f in files:
        for r in bc.read_csv_rows(f):
            merged[(r["complex_id"], r["arm"], str(r["k"]), str(r.get("pose", "")))] = r
    rows = list(merged.values())
    bc.write_csv_rows(P["docking"], rows, DOCK_FIELDS)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"merged {len(files)} shard file(s) -> {len(rows)} rows ({ok} ok)")
    print(f"docking_summary.csv: {P['docking']}")
    bc.write_manifest(root, "docking_merge", {
        "n_shard_files": len(files), "n_rows": len(rows), "n_ok": ok,
        "docking_summary_sha256": bc.sha256_file(P["docking"]),
    })
    return 0


def run(args: argparse.Namespace) -> int:
    root = bc.bundle_dir(args.out_dir)
    P = bc.paths(root)
    if args.merge_shards:
        return merge_shards(root)
    if not os.path.exists(P["predictions"]):
        print(f"ERROR: no predictions.csv at {P['predictions']}. Run generate_predictions.py first.")
        return 2

    jobs = collect_jobs(root, args.complex or None)
    if args.num_shards > 1:
        mine = set(select_shard(list(jobs), args.shard, args.num_shards))
        jobs = {c: j for c, j in jobs.items() if c in mine}
        os.makedirs(shard_dir(root), exist_ok=True)
        log(f"shard {args.shard}/{args.num_shards}: {len(jobs)} complexes "
            f"-> {os.path.basename(shard_path(root, args.shard))}")
        if not jobs:
            print("nothing for this shard (more tasks than complexes); exiting 0.")
            return 0
    # Where THIS process writes. A sharded task owns its own file; an unsharded
    # run owns docking_summary.csv, as before.
    out_csv = shard_path(root, args.shard) if args.num_shards > 1 else P["docking"]

    n_lig = sum(len(j["ligands"]) for j in jobs.values())
    log(f"{len(jobs)} complexes, {n_lig} ligands to dock "
        f"(including one parent redock each)")

    gnina_bin = args.gnina_binary or getattr(bc.config, "GNINA_BINARY", "gnina")
    if not args.prepare_only:
        if platform.system() != "Linux":
            print(f"\nGNINA is Linux-only (Stage 6 A1) and this is {platform.system()}.\n"
                  f"Preparing inputs only; run this script again on the cluster to dock.\n")
            args.prepare_only = True
        else:
            import stage6_1_docking as s6
            if not os.path.exists(gnina_bin):
                print(f"\nERROR: no GNINA at {gnina_bin}.\n"
                      f"       Compute nodes cannot download it (and it is 1.4-2.1 GB).\n"
                      f"       From a LOGIN node:  bash baseline_analysis/get_gnina.sh\n")
                return 5
            s6._make_executable(gnina_bin)
            log(f"GNINA: {gnina_bin}")
            log(f"       {gnina_version(gnina_bin)}")

    existing: Dict[tuple, dict] = {}
    if os.path.exists(out_csv) and not args.overwrite:
        for r in bc.read_csv_rows(out_csv):
            existing[(r["complex_id"], r["arm"], str(r["k"]), str(r.get("pose", "")))] = r
        log(f"existing docking rows in {os.path.basename(out_csv)}: {len(existing)}")

    rows: List[dict] = []
    done_complexes = 0
    for cid, job in sorted(jobs.items()):
        rows_c = dock_complex(root, cid, job["meta"], job["ligands"], gnina_bin, args)
        rows.extend(rows_c)
        done_complexes += 1
        ok = sum(1 for r in rows_c if r.get("status") == "ok")
        log(f"  [{done_complexes}/{len(jobs)}] {cid}: {ok} scored rows"
            + ("" if not any(r.get("status") == "error" for r in rows_c)
               else f", {sum(1 for r in rows_c if r.get('status') == 'error')} errors"))
        merged = dict(existing)
        for r in rows:
            merged[(r["complex_id"], r["arm"], str(r["k"]), str(r.get("pose", "")))] = r
        bc.write_csv_rows(out_csv, list(merged.values()), DOCK_FIELDS)

    merged = dict(existing)
    for r in rows:
        merged[(r["complex_id"], r["arm"], str(r["k"]), str(r.get("pose", "")))] = r
    final = list(merged.values())
    bc.write_csv_rows(out_csv, final, DOCK_FIELDS)

    # A sharded task must not touch the shared manifest either: 24 concurrent
    # read-modify-write cycles on one JSON lose sections. The merge step records
    # the run instead.
    if args.num_shards <= 1:
        bc.write_manifest(root, "docking", {
            "gnina_binary": gnina_bin,
            "gnina_version": gnina_version(gnina_bin) if not args.prepare_only else "",
            "sdf_properties_seen": sorted(PROPS_SEEN),
            "num_modes": args.num_modes, "exhaustiveness": args.exhaustiveness,
            "seed": args.seed, "prepare_only": bool(args.prepare_only),
            "n_complexes": len(jobs), "n_ligands": n_lig,
            "n_rows": len(final),
            "docking_summary_sha256": bc.sha256_file(P["docking"]),
        })

    ok = [r for r in final if r.get("status") == "ok"]
    print(f"\ndocking rows: {len(final)} ({len(ok)} ok)   -> {out_csv}")

    # A GNINA whose SDF carries none of the expected score properties would
    # otherwise produce a full table of blanks and an analysis that reports
    # "no docking data". Say so here, with what the file actually contained.
    scored = [r for r in ok if any(str(r.get(k, "")).strip() for k in _SCORE_ALIASES)]
    if ok and not scored:
        print("\n*** WARNING: docking produced poses but NO recognised score property.")
        print(f"    Properties found in the SDFs: {sorted(PROPS_SEEN) or '(none)'}")
        print(f"    Expected one of: {sorted(_SCORE_ALIASES)}")
        print("    Check the GNINA version; add the new spelling to _SCORE_ALIASES.")
    elif PROPS_SEEN:
        log(f"SDF properties seen: {', '.join(sorted(PROPS_SEEN))}")
    rm = [float(r["redock_rmsd"]) for r in final
          if r.get("arm") == PARENT_REDOCK and str(r.get("redock_rmsd") or "").strip()
          and str(r.get("pose")) == "1"]
    if rm:
        good = sum(1 for x in rm if x <= 2.0)
        print(f"redock RMSD: {good}/{len(rm)} complexes within 2.0 A "
              f"(median {sorted(rm)[len(rm)//2]:.2f} A) -- the docking-setup sanity check")
    return 0


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST  (mock GNINA -- no binary, no GPU, runs on any platform)
# ════════════════════════════════════════════════════════════════════════════

def _self_test() -> int:
    import tempfile

    from rdkit import Chem
    from rdkit.Chem import AllChem

    print("run_docking.py self-test")

    cmd = gnina_cmd("/bin/gnina", "rec.pdb", "lig.sdf", "orig.pdb", "out.sdf",
                    "d.log", 9, 8, 42)
    assert "--autobox_ligand" in cmd and "--seed" in cmd and "--exhaustiveness" in cmd
    assert cmd[cmd.index("--seed") + 1] == "42"
    sc = gnina_cmd("/bin/gnina", "rec.pdb", "x.sdf", "orig.pdb", "", "d.log", 9, 8, 42,
                   score_only=True)
    assert "--score_only" in sc and "--autobox_ligand" not in sc and "--out" not in sc
    print("  ✓ gnina_cmd: docking pins --seed/--exhaustiveness; --score_only omits the search")

    # Score parsing must survive GNINA renaming a property, and must SAY when it
    # cannot find one -- a silent "" for every pose is the failure that produces
    # an empty analysis rather than an error.
    with tempfile.TemporaryDirectory() as td:
        from rdkit import Chem as _C
        from rdkit.Chem import AllChem as _A

        def _sdf(path, props):
            m = _C.AddHs(_C.MolFromSmiles("CCO"))
            _A.EmbedMolecule(m, _A.ETKDGv3())
            w = _C.SDWriter(path)
            for p in props:
                mm = _C.Mol(m)
                for k, v in p.items():
                    mm.SetProp(k, str(v))
                w.write(mm)
            w.close()

        p1 = os.path.join(td, "v10.sdf")
        _sdf(p1, [{"CNNscore": 0.9, "CNNaffinity": 6.9, "minimizedAffinity": -9.1}])
        rows_, seen_ = parse_gnina_scores(p1)
        assert rows_[0]["CNNaffinity"] == 6.9 and rows_[0]["minimizedAffinity"] == -9.1
        assert "CNNscore" in seen_
        print("  ✓ parse_gnina_scores: v1.0-style properties read, names reported back")

        p2 = os.path.join(td, "alias.sdf")
        _sdf(p2, [{"CNN_affinity": 7.2, "Affinity": -8.8}])
        rows2_, _ = parse_gnina_scores(p2)
        assert rows2_[0]["CNNaffinity"] == 7.2, rows2_
        assert rows2_[0]["minimizedAffinity"] == -8.8, rows2_
        print("  ✓ parse_gnina_scores: alternative spellings (CNN_affinity / Affinity) mapped")

        p3 = os.path.join(td, "unknown.sdf")
        _sdf(p3, [{"SomeNewScore": 1.0}])
        rows3_, seen3_ = parse_gnina_scores(p3)
        assert rows3_ and all(rows3_[0][k] == "" for k in _SCORE_ALIASES)
        assert "SomeNewScore" in seen3_
        print("  ✓ parse_gnina_scores: unrecognised properties -> empty scores + the "
              "names surfaced (run() turns this into a loud warning)")

    v = gnina_version(os.path.join(os.sep, "definitely", "not", "gnina"))
    assert v.startswith("(unavailable"), v
    print(f"  ✓ gnina_version: missing binary -> {v!r}, recorded not raised")

    got = parse_score_only("CNNscore: 0.812\nCNNaffinity 6.44\nAffinity: -8.3 (kcal/mol)\n")
    assert got == {"CNNscore": "0.812", "CNNaffinity": "6.44", "minimizedAffinity": "-8.3"}, got
    assert parse_score_only("nothing here")["CNNscore"] == ""
    print("  ✓ parse_score_only: reads all three scores; missing -> '' not 0")

    with tempfile.TemporaryDirectory() as td:
        # A real 3-atom "complex": receptor ATOM lines + ligand HETATM lines.
        pdb = os.path.join(td, "complex.pdb")
        with open(pdb, "w") as f:
            for i in range(1, 4):
                f.write(f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                        f"{i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n")
            for j, (nm, el, x) in enumerate([("C1", "C", 10.0), ("O1", "O", 11.2)], start=10):
                f.write(f"HETATM{j:5d}  {nm:<3s} LIG A 600    "
                        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           {el}\n")
            f.write("END\n")

        import stage6_1_docking as s6
        rec, orig = os.path.join(td, "rec.pdb"), os.path.join(td, "orig.pdb")
        assert s6.prepare_receptor(pdb, rec)
        assert s6.prepare_autobox_ligand(pdb, orig, "LIG", "A", 600)
        assert sum(1 for ln in open(rec) if ln.startswith("ATOM")) == 3
        assert sum(1 for ln in open(orig) if ln.startswith("HETATM")) == 2
        print("  ✓ receptor/autobox prep: 3 ATOM records -> rec.pdb, 2 HETATM -> orig.pdb")

        crystal = crystal_sdf_from_pdb(orig, "CO", os.path.join(td, "crystal.sdf"))
        print(f"  {'✓' if crystal else '-'} crystal.sdf from HETATM + parent SMILES "
              f"({'written' if crystal else 'not derivable for this stub, treated as optional'})")

        # A canned pose file lets a mock GNINA stand in for the real binary.
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(m)
        m.SetProp("_Name", "pose")
        canned = os.path.join(td, "canned.sdf")
        w = Chem.SDWriter(canned)
        for pose, (cnn, aff, vina) in enumerate([(0.91, 6.9, -9.1), (0.72, 6.1, -8.2)], 1):
            mm = Chem.Mol(m)
            mm.SetProp("CNNscore", str(cnn))
            mm.SetProp("CNNaffinity", str(aff))
            mm.SetProp("minimizedAffinity", str(vina))
            w.write(mm)
        w.close()

        calls: List[List[str]] = []

        def mock_runner(cmd_: List[str]) -> Tuple[int, str, str]:
            calls.append(cmd_)
            if "--score_only" in cmd_:
                return 0, "CNNscore: 0.88\nCNNaffinity 7.10\nAffinity: -9.9\n", ""
            out = cmd_[cmd_.index("--out") + 1]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(canned) as src, open(out, "w") as dst:
                dst.write(src.read())
            open(cmd_[cmd_.index("--log") + 1], "w").write("mock\n")
            return 0, "", ""

        opts = argparse.Namespace(num_modes=9, exhaustiveness=8, seed=42, timeout=60,
                                  prepare_only=False, no_crystal_score=False)
        meta = {"uniprot": "P0", "fda_status": "4.0", "parent_smiles": "CCO", "pdb_path": pdb}
        ligands = [(PARENT_REDOCK, "", "CCO"), ("plip_pos", "0", "CCN"), ("random", "0", "CCC")]
        root = os.path.join(td, "bundle")
        rows = dock_complex(root, "1TST:LIG:A:600", meta, ligands, "/bin/mock", opts,
                            runner=mock_runner)

        arms = {r["arm"] for r in rows}
        assert PARENT_REDOCK in arms and PARENT_CRYSTAL in arms, arms
        assert all(r["status"] == "ok" for r in rows), [r for r in rows if r["status"] != "ok"]
        per_pose = [r for r in rows if r["arm"] == "plip_pos"]
        assert len(per_pose) == 2 and {r["pose"] for r in per_pose} == {1, 2}
        assert per_pose[0]["CNNaffinity"] == 6.9
        print(f"  ✓ dock_complex: {len(rows)} rows across {len(arms)} arms "
              f"(parent_crystal + parent_redock + predictions), 2 poses each")

        n_before = len(calls)
        rows2 = dock_complex(root, "1TST:LIG:A:600", meta, ligands, "/bin/mock", opts,
                             runner=mock_runner)
        docking_calls = [c for c in calls[n_before:] if "--score_only" not in c]
        assert not docking_calls, docking_calls
        assert len(rows2) == len(rows)
        print("  ✓ resume: re-running re-docks nothing (existing poses reused), same rows returned")

        # pose_rmsd on a controlled pair: the same molecule translated by a known
        # amount must come back as that distance, and a different molecule as None.
        ref_m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(ref_m, AllChem.ETKDGv3())
        ref_p, moved_p = os.path.join(td, "r.sdf"), os.path.join(td, "m.sdf")
        wr = Chem.SDWriter(ref_p); wr.write(ref_m); wr.close()
        moved = Chem.Mol(ref_m)
        conf = moved.GetConformer()
        for i in range(moved.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (pos.x + 1.0, pos.y, pos.z))
        wm = Chem.SDWriter(moved_p); wm.write(moved); wm.close()
        d_same = pose_rmsd(ref_p, ref_p)
        d_moved = pose_rmsd(ref_p, moved_p)
        assert d_same == 0.0, d_same
        assert d_moved is not None and abs(d_moved - 1.0) < 0.05, d_moved
        other_p = os.path.join(td, "o.sdf")
        wo = Chem.SDWriter(other_p); wo.write(Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))); wo.close()
        assert pose_rmsd(ref_p, other_p) is None
        print(f"  ✓ pose_rmsd: identical pose -> {d_same} A, 1 A translation -> {d_moved} A, "
              f"mismatched molecule -> None")

        r1 = [r for r in rows if r["arm"] == PARENT_REDOCK and r["pose"] == 1][0]
        print(f"  {'✓' if r1['redock_rmsd'] != '' else '-'} redock RMSD recorded on the "
              f"parent_redock pose-1 row: {r1['redock_rmsd'] if r1['redock_rmsd'] != '' else 'n/a for this stub'}")

        opts.prepare_only = True
        prep = dock_complex(root, "1TST:LIG:A:600", meta, ligands, "/bin/mock", opts,
                            runner=mock_runner)
        assert len(prep) == 1 and prep[0]["status"] == "prepared"
        print("  ✓ --prepare-only: inputs written, no docking attempted")

        bad = dock_complex(root, "9XXX:LIG:A:1", {"pdb_path": os.path.join(td, "nope.pdb")},
                           ligands, "/bin/mock", opts, runner=mock_runner)
        assert bad[0]["status"] == "error" and "complex.pdb" in bad[0]["note"]
        print("  ✓ missing structure -> one error row, run continues")

    # collect_jobs dedupes and adds the reference.
    with tempfile.TemporaryDirectory() as td:
        P = bc.paths(td)
        preds = []
        for arm in bc.ARMS:
            for k in range(2):
                preds.append({"complex_id": "1TST:LIG:A:600", "arm": arm, "k": k,
                              "parent_smiles": "CCO", "generated_smiles": "CCN",
                              "valid": 1.0, "uniprot": "P0", "fda_status": "4.0"})
        preds.append({"complex_id": "1TST:LIG:A:600", "arm": "random", "k": 9,
                      "parent_smiles": "CCO", "generated_smiles": "", "valid": 0.0})
        import generate_predictions as gp
        bc.write_csv_rows(P["predictions"], preds, gp.PRED_FIELDS)
        jobs = collect_jobs(td, None)
        ligs = jobs["1TST:LIG:A:600"]["ligands"]
        assert ligs[0][0] == PARENT_REDOCK and ligs[0][2] == "CCO"
        assert len(ligs) == 2, ligs        # parent + one distinct valid SMILES
        print(f"  ✓ collect_jobs: 6 valid predictions of one molecule + 1 invalid "
              f"-> {len(ligs)} docking jobs (parent redock + deduped prediction)")

    # Sharding: every complex owned exactly once, and shards merge cleanly.
    ids = [f"{i}ABC:LIG:A:1" for i in range(24)]
    shards = [select_shard(ids, i, 7) for i in range(7)]
    flat = [c for s in shards for c in s]
    assert sorted(flat) == sorted(ids), "sharding lost or duplicated a complex"
    assert len(set(map(len, shards))) <= 2, [len(s) for s in shards]
    assert select_shard(ids, 0, 7) == select_shard(ids, 0, 7), "sharding is not stable"
    assert select_shard(ids, 5, 30) == [] or len(select_shard(ids, 5, 30)) <= 1
    print(f"  ✓ select_shard: 24 complexes over 7 tasks -> sizes {[len(s) for s in shards]}, "
          f"union == input, stable across calls")

    with tempfile.TemporaryDirectory() as td:
        P = bc.paths(td)
        os.makedirs(shard_dir(td), exist_ok=True)
        for i in range(3):
            bc.write_csv_rows(shard_path(td, i), [
                {"complex_id": f"{i}ABC:LIG:A:1", "arm": PARENT_REDOCK, "k": "", "pose": "1",
                 "smiles": "CCO", "CNNaffinity": f"{6 + i}", "status": "ok"},
                {"complex_id": f"{i}ABC:LIG:A:1", "arm": "plip_pos", "k": "0", "pose": "1",
                 "smiles": "CCN", "CNNaffinity": f"{7 + i}", "status": "ok"},
            ], DOCK_FIELDS)
        assert merge_shards(td) == 0
        rows = bc.read_csv_rows(P["docking"])
        assert len(rows) == 6, len(rows)
        assert len({r["complex_id"] for r in rows}) == 3
        # Re-merging must be idempotent, not duplicate every row.
        merge_shards(td)
        assert len(bc.read_csv_rows(P["docking"])) == 6
        print("  ✓ merge_shards: 3 shard files -> 6 rows in docking_summary.csv, "
              "re-merging is idempotent")

    print("\nALL SELF-TESTS PASSED")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    cfg = bc.config
    p = argparse.ArgumentParser(description="Dock the parent and every prediction with GNINA.")
    p.add_argument("--out-dir", default=None, help="Bundle root (default config.BASELINE_DIR).")
    p.add_argument("--complex", action="append", default=None,
                   help="Only this complex id (repeatable), e.g. --complex 1ERR:RAL:B:600.")
    p.add_argument("--gnina-binary", default=None, help="GNINA path (default config.GNINA_BINARY).")
    p.add_argument("--num-modes", type=int, default=getattr(cfg, "BASELINE_GNINA_NUM_MODES", 9))
    p.add_argument("--exhaustiveness", type=int,
                   default=getattr(cfg, "BASELINE_GNINA_EXHAUSTIVENESS", 8))
    p.add_argument("--seed", type=int, default=getattr(cfg, "BASELINE_GNINA_SEED", 42))
    p.add_argument("--timeout", type=int, default=3600, help="Per-GNINA-call timeout (s).")
    p.add_argument("--prepare-only", action="store_true", help="Write inputs, do not dock.")
    p.add_argument("--no-crystal-score", action="store_true",
                   help="Skip the --score_only pass on the crystal pose.")
    p.add_argument("--overwrite", action="store_true", help="Ignore an existing docking_summary.csv.")
    p.add_argument("--shard", type=int, default=0,
                   help="This task's index (0-based) when running as a SLURM array.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total array tasks. >1 makes each task write its own "
                        "docking/_shards/docking_shard_<i>.csv instead of the shared summary.")
    p.add_argument("--merge-shards", action="store_true",
                   help="Fold every docking/_shards/*.csv into docking_summary.csv and exit. "
                        "Run once, after the array finishes.")
    p.add_argument("--test", action="store_true", help="Run the self-test and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.test:
        return _self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""
stage6_docking.py
=================
Stage 6 of the pipeline — molecular docking + score visualisation.

Uses GNINA (v1.0.3) with autobox to dock all generated molecules from
Stage 2 into the BRD4 binding pocket defined by the original ligand
in each PDB structure.

Per ligand group (e.g. EAM-A-1), produces:
  <stage6_dir>/<group>/
      rec.pdb                       — receptor (ATOM records only)
      orig.pdb                      — autobox ligand (HETATM)
      mol_0001/
          ligand.sdf                — 3D conformer input to GNINA
          docked_poses.sdf          — GNINA multi-pose output
          complex_pose001.pdb       — receptor + docked ligand (up to n_poses)
          docking.log
      mol_0002/ ...

  docking_summary.csv               — one row per (group, molecule, pose)
      columns: ligand_group, mol_idx, smiles, pose, CNNscore,
               CNNaffinity, minimizedAffinity, complex_pdb

After docking, two visualisation PNGs are saved to STAGE6_DIR:
  stage6_docking_top10_molecules_all_scores_BARCHART.png
  stage6_docking_all_molecules_all_scores_BARCHART.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1.  GNINA is Linux-only (x86_64). The auto-download fetches the
     pre-compiled binary from GitHub. This works in Colab and on any
     Linux machine. macOS / Windows require WSL or manual compilation.

A2.  GNINA binary location is config.GNINA_BINARY.
     If not found, Stage 6 auto-downloads it.
     The download URL is config.GNINA_DOWNLOAD_URL.
     Google Drive (Colab FUSE) cannot execute binaries; the binary is
     copied to /content/gnina before every run.

A3.  Receptor preparation: ATOM records from the PDB are written to
     rec.pdb. HETATM records matching resname + chain + resseq are
     written to orig.pdb (autobox ligand).

A4.  SMILES → 3D SDF conversion uses RDKit ETKDGv3 + UFF minimisation.
     If 3D embedding fails the molecule is skipped with a warning.

A5.  The PDB file path and ligand identity for each group are read from
     Stage-1 .meta.json files in config.MASK_CALC_OUTDIR, matched by
     the key "{resname}-{chain}-{resseq}".

A6.  All generated molecules are docked (no pre-filter).

A7.  Number of poses to keep as complex PDB files is asked at runtime
     (default 1). GNINA always generates up to 9 internally; the CSV
     always contains all GNINA poses regardless of this setting.

A8.  Score properties read from GNINA's output SDF:
       CNNscore          — CNN-predicted binding probability  (higher = better)
       CNNaffinity       — CNN-predicted −log Kd/Ki           (higher = better)
       minimizedAffinity — Vinardo score (kcal/mol)           (more negative = better)
     If a property is absent in a pose, it is written as "" in the CSV.

A9.  The complex PDB writing code is transcribed verbatim from the
     reference notebook (Cell 12) to avoid any import dependency.

A10. The self-test creates a mock GNINA shell script that copies a
     pre-built minimal SDF to the expected output path so no real
     binary or GPU is required.

A11. already_docked cache: a molecule is considered already docked if
     its docked_poses.sdf exists, is non-empty, AND RDKit can parse at
     least one pose from it. Cached molecules are added to the CSV but
     GNINA is not called again. n_cached is tracked and reported.

A12. Visualisation ranking: CNNaffinity is the primary sort key
     (higher = better); minimizedAffinity is the tiebreaker
     (lower = better). Only pose 1 per molecule is used for ranking.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN  (Colab)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python stage6_docking.py
  GNINA is downloaded automatically on first run (~500 MB).

HOW TO RUN  (local Linux)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  wget https://github.com/gnina/gnina/releases/download/v1.0.3/gnina
  chmod +x gnina   # then set config.GNINA_BINARY to its absolute path
  python stage6_docking.py

HOW TO TEST  (no GNINA, no GPU required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Answer "yes" to the smoke-test prompt when running the file.
  Writes to config.STAGE6_DIR/test/ (wiped each run).
"""

import csv
import glob
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import urllib.request
import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe in Colab and headless
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import config
from stage4_br4_matching import _safe, load_pool_for_ligand


# ════════════════════════════════════════════════════════════════════════════
#  GNINA BINARY MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def _download_gnina(dest_path: str, url: str) -> None:
    """Download the GNINA binary to dest_path and make it executable."""
    # Guard: a previous failed run may have left a directory at dest_path.
    if os.path.isdir(dest_path):
        print(f"  ⚠️  Found a directory at '{dest_path}' — removing it "
               "(left by a previous failed download attempt).")
        shutil.rmtree(dest_path)
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    print(f"  ⬇️  Downloading GNINA from:\n    {url}")
    print(f"  Destination: {dest_path}")
    print("  (This is a ~500 MB binary — may take 1–5 minutes on a slow connection.)")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(block_num * block_size / total_size * 100, 100)
            if block_num % 200 == 0:
                print(f"    {pct:.0f}% ...", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    current = os.stat(dest_path).st_mode
    os.chmod(dest_path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  ✅ GNINA downloaded and made executable: {dest_path}")


# Local execution path: Google Drive (Colab FUSE) cannot execute binaries.
# We always copy the Drive binary here before running it.
_GNINA_LOCAL = "/content/gnina"


def _make_executable(path: str) -> None:
    current = os.stat(path).st_mode
    os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def ensure_gnina(
    binary_path: str = config.GNINA_BINARY,
    download_url: str = config.GNINA_DOWNLOAD_URL,
) -> str:
    """
    Return the path to a locally executable GNINA binary.

    1. If config.GNINA_BINARY exists on Drive → copy to _GNINA_LOCAL.
    2. If missing → download to Drive first, then copy to _GNINA_LOCAL.

    The Drive copy is the permanent store; _GNINA_LOCAL is the
    session-local executable (required because Drive is FUSE-mounted
    in Colab and the kernel cannot execve() FUSE-backed paths).
    """
    drive_path = binary_path
    local_path = _GNINA_LOCAL

    if not os.path.isfile(drive_path):
        print(f"  GNINA not found at '{drive_path}'. Downloading to Drive ...")
        _download_gnina(drive_path, download_url)
    else:
        print(f"  ✅ GNINA found on Drive: {drive_path}")

    drive_mtime = os.path.getmtime(drive_path)
    local_ok = (
        os.path.isfile(local_path)
        and os.path.getmtime(local_path) >= drive_mtime
        and os.access(local_path, os.X_OK)
    )
    if not local_ok:
        print(f"  📋 Copying GNINA to local path for execution: {local_path}")
        shutil.copy2(drive_path, local_path)
        _make_executable(local_path)
        print(f"  ✅ GNINA ready at local path: {local_path}")
    else:
        print(f"  ✅ GNINA already cached locally: {local_path}")

    return local_path


# ════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PROMPTS
# ════════════════════════════════════════════════════════════════════════════

def _ask_yes_no_default(question: str, default: bool) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"  {question} {hint}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("yes", "y"):
            return True
        if raw in ("no", "n"):
            return False
        print("    Please type 'yes'/'y' or 'no'/'n', or press Enter for the default.")


def _ask_n_poses() -> int:
    print("""
  Number of docking poses to write as complex PDB files.
    GNINA generates up to 9 poses internally; this controls how many
    are saved as receptor+ligand PDB files.
    The score CSV always contains all poses regardless of this setting.
    1 = best pose only  [default]
    9 = all poses
""")
    while True:
        raw = input("  Number of poses to keep (1–9) [1]: ").strip()
        if raw == "":
            return 1
        try:
            val = int(raw)
            if 1 <= val <= 9:
                return val
        except ValueError:
            pass
        print("    Please type an integer between 1 and 9.")


def _ask_generator_stage() -> str:
    """
    Ask which generator stage produced the molecules to dock.
    Returns one of: "2", "2.5", "2.7"

    Maps to:
      2   → config.PRED_DIR          (Stage 2 prefix-slice)
      2.5 → config.STAGE25_PRED_DIR  (Stage 2.5 random-pick, single seed)
      2.7 → config.STAGE27_PRED_DIR  (Stage 2.7 multi-seed aggregated)
    """
    print(f"""
  Generator stage — which predicted molecules do you want to dock?

    2   : Stage 2   — prefix-slice masking
          {config.PRED_DIR}
    2.5 : Stage 2.5 — random-pick, single seed
          {config.STAGE25_PRED_DIR}
    2.7 : Stage 2.7 — multi-seed aggregated
          {config.STAGE27_PRED_DIR}
""")
    valid = {"2", "2.5", "2.7"}
    while True:
        raw = input("  Generator stage (2 / 2.5 / 2.7) [2]: ").strip()
        if raw == "":
            return "2"
        if raw in valid:
            return raw
        print("    Please type 2, 2.5, or 2.7.")


def _resolve_pred_dir(gen_stage: str) -> str:
    """Return the predictions directory for the given generator stage."""
    return {
        "2":   config.PRED_DIR,
        "2.5": config.STAGE25_PRED_DIR,
        "2.7": config.STAGE27_PRED_DIR,
    }[gen_stage]


def _nested_stage6_dir(gen_stage: str, pool: str) -> str:
    """
    Build the nested output directory for one docking run.

    Structure:
      config.STAGE6_DIR / stage{gen_stage} / {pool} /

    Example with expo05:
      .../expo05/stage6_docking/stage2.7/ia/
    """
    return os.path.join(
        config.STAGE6_DIR,
        f"stage{gen_stage}",
        pool,
    )


def ask_runtime_options() -> int:
    """Ask all Stage-6 runtime options. Returns n_poses."""
    print("\n" + "─" * 60)
    print("STAGE 6 OPTIONS")
    print("─" * 60)
    n_poses = _ask_n_poses()
    print(f"\n  Settings confirmed:")
    print(f"    Poses to keep: {n_poses}")
    print("─" * 60 + "\n")
    return n_poses


# ════════════════════════════════════════════════════════════════════════════
#  MOLECULAR PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def smiles_to_sdf(smiles: str, out_path: str, name: str = "Ligand") -> bool:
    """
    Convert SMILES to a 3D SDF using RDKit ETKDGv3 + UFF.
    Returns True on success, False if embedding fails (Assumption A4).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"    ⚠️  Cannot parse SMILES: {smiles[:60]}")
        return False
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1:
        print(f"    ⚠️  3D embedding failed for: {smiles[:60]}")
        return False
    AllChem.UFFOptimizeMolecule(mol)
    mol.SetProp("_Name", name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    w = Chem.SDWriter(out_path)
    w.write(mol)
    w.close()
    return True


def prepare_receptor(pdb_path: str, rec_out: str) -> bool:
    """Write ATOM records from pdb_path to rec_out (receptor only)."""
    os.makedirs(os.path.dirname(rec_out) or ".", exist_ok=True)
    lines = [ln for ln in open(pdb_path)
             if ln.startswith("ATOM  ") or ln.startswith("ATOM ")]
    if not lines:
        print(f"    ⚠️  No ATOM records found in {pdb_path}")
        return False
    with open(rec_out, "w") as f:
        f.writelines(lines)
        f.write("END\n")
    return True


def prepare_autobox_ligand(
    pdb_path: str, orig_out: str,
    resname: str, chain: str, resseq: int,
) -> bool:
    """Write HETATM records for the autobox reference ligand."""
    os.makedirs(os.path.dirname(orig_out) or ".", exist_ok=True)
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip() != resname:
                continue
            if line[21].strip() != chain:
                continue
            try:
                if int(line[22:26]) != resseq:
                    continue
            except ValueError:
                continue
            lines.append(line)
    if not lines:
        print(f"    ⚠️  No HETATM records for {resname}:{chain}:{resseq} in {pdb_path}")
        return False
    with open(orig_out, "w") as f:
        f.writelines(lines)
        f.write("END\n")
    return True


# ════════════════════════════════════════════════════════════════════════════
#  DOCKING
# ════════════════════════════════════════════════════════════════════════════

def run_gnina(
    gnina_bin: str, receptor: str, ligand: str,
    autobox: str, out_sdf: str, log_file: str,
    n_poses: int = 9,
) -> bool:
    """Run GNINA docking via subprocess. Returns True on success."""
    os.makedirs(os.path.dirname(out_sdf) or ".", exist_ok=True)
    cmd = [
        gnina_bin,
        "-r", receptor, "-l", ligand,
        "--autobox_ligand", autobox,
        "--out", out_sdf, "--log", log_file,
        "--num_modes", str(n_poses),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ⚠️  GNINA failed (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else "  (no stderr)")
        return False
    return True


# ════════════════════════════════════════════════════════════════════════════
#  SCORE PARSING
# ════════════════════════════════════════════════════════════════════════════

_SCORE_PROPS = ("CNNscore", "CNNaffinity", "minimizedAffinity")


def parse_gnina_scores(sdf_path: str) -> List[Dict[str, str]]:
    """
    Read GNINA output SDF and extract per-pose scores.
    Returns list of dicts, one per pose.
    """
    if not os.path.exists(sdf_path):
        return []
    rows, pose = [], 0
    for mol in Chem.ForwardSDMolSupplier(sdf_path, removeHs=False, sanitize=False):
        if mol is None:
            continue
        pose += 1
        row = {"pose": pose}
        for prop in _SCORE_PROPS:
            row[prop] = mol.GetPropsAsDict().get(prop, "")
        rows.append(row)
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  COMPLEX WRITING — transcribed verbatim from notebook Cell 12 (Assumption A9)
# ════════════════════════════════════════════════════════════════════════════

_TWO_LETTER = {
    "BR", "CL", "FE", "ZN", "MG", "NA", "CA", "MN",
    "CO", "NI", "CU", "AL", "SI", "KR", "XE", "SR",
    "CD", "HG", "PB", "SN",
}


def _ensure_nl80(line: str) -> str:
    if not line.endswith("\n"):
        line += "\n"
    if len(line) < 81:
        line = line[:-1] + " " * (81 - len(line)) + "\n"
    else:
        line = line[:80] + "\n"
    return line


def _split_before_terminal(records: List[str]) -> Tuple[List[str], List[str]]:
    terminal = {"END", "ENDMDL", "MASTER"}
    i = len(records)
    while i > 0:
        tok = records[i - 1].strip().split()[:1]
        if tok and tok[0] in terminal:
            i -= 1
        else:
            break
    return records[:i], records[i:]


def _last_serial(lines: List[str]) -> int:
    mx = 0
    for ln in lines:
        if ln.startswith(("ATOM  ", "HETATM")):
            try:
                mx = max(mx, int(ln[6:11]))
            except Exception:
                pass
    return mx


def _infer_element_from_name(name: str) -> str:
    s = "".join(ch for ch in name.strip() if ch.isalpha()).upper()
    if len(s) >= 2 and s[:2] in _TWO_LETTER:
        return s[:2]
    return s[:1] or "C"


def _format_atom_name(name: str, elem: str) -> str:
    e = (elem or "").strip().upper()
    n = (name or e).strip()
    return f"{n:>4}"[:4] if len(e) == 1 else f"{n:<4}"[:4]


def _format_hetatm(
    serial: int, elem: str, atom_name: str,
    resname: str, chain: str, resseq: int,
    x: float, y: float, z: float,
    occ: float = 1.00, bfac: float = 0.00,
    alt: str = " ", icode: str = " ",
) -> str:
    e = (elem or "").strip().upper() or _infer_element_from_name(atom_name)
    aname = _format_atom_name(atom_name, e)
    res   = f"{(resname or 'LIG').upper():>3}"[:3]
    ch    = (chain or " ")[:1]
    line  = (
        f"HETATM{serial:5d} {aname}{alt[:1] if alt else ' '}"
        f"{res} {ch}{int(resseq):4d}{icode[:1] if icode else ' '}"
        f"   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occ:6.2f}{bfac:6.2f}"
        f"{'':10}{e:>2}{'':2}"
    )
    return _ensure_nl80(line)


def _load_poses_robust(sdf_path: str) -> Iterable[Chem.Mol]:
    """Yield RDKit Mol per pose; fall back to bond-order determination on failure."""
    from rdkit.Chem.rdDetermineBonds import DetermineConnectivity, DetermineBondOrders
    fsup = Chem.ForwardSDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    for m in fsup:
        if m is None:
            continue
        m.UpdatePropertyCache(strict=False)
        try:
            Chem.SanitizeMol(m)
            yield m
            continue
        except Exception:
            pass
        try:
            DetermineConnectivity(m)
            total_charge = sum(a.GetFormalCharge() for a in m.GetAtoms())
            DetermineBondOrders(
                m, charge=total_charge,
                allowChargedFragments=True, embedChiral=True, useAtomMap=False,
            )
            Chem.SanitizeMol(m)
            yield m
        except Exception:
            yield m


def write_complexes_from_gnina_sdf(
    receptor_pdb: str, poses_sdf: str, out_prefix: str,
    ligand_resname: str = "LIG", chain_id: str = "A", resseq: int = 1,
    write_conect: bool = True, unique_atom_names: bool = True,
    max_poses: int = 9,
) -> List[str]:
    """
    Create one complex PDB per pose: receptor + ligand HETATM (+ CONECT).
    max_poses caps the number of complex PDBs written (Assumption A7).
    """
    with open(receptor_pdb) as fh:
        rec_all = fh.readlines()
    rec_prefix, rec_terminal = _split_before_terminal(rec_all)
    base_serial = _last_serial(rec_prefix)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    written:  List[str] = []
    pose_idx = 0

    for mol in _load_poses_robust(poses_sdf):
        if mol.GetNumAtoms() == 0:
            continue
        pose_idx += 1
        if pose_idx > max_poses:
            break

        conf   = mol.GetConformer()
        serial = base_serial
        het_lines:  List[str] = []
        idx2serial: Dict[int, int] = {}
        elem_counts: Dict[str, int] = defaultdict(int)

        def unique_name(sym: str) -> str:
            s = (sym or "X").upper()
            elem_counts[s] += 1
            return f"{s}{elem_counts[s]}"[:4]

        for a in mol.GetAtoms():
            i    = a.GetIdx()
            p    = conf.GetAtomPosition(i)
            elem = a.GetSymbol() or "C"
            name = unique_name(elem) if unique_atom_names else (elem or "X")
            serial += 1
            het_lines.append(
                _format_hetatm(
                    serial=serial, elem=elem, atom_name=name,
                    resname=ligand_resname, chain=chain_id, resseq=int(resseq),
                    x=p.x, y=p.y, z=p.z,
                )
            )
            idx2serial[i] = serial

        conect_lines: List[str] = []
        if write_conect:
            neigh: Dict[int, List[int]] = defaultdict(list)
            for b in mol.GetBonds():
                i, j   = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                si, sj = idx2serial.get(i), idx2serial.get(j)
                if si is None or sj is None:
                    continue
                neigh[si].append(sj)
                neigh[sj].append(si)
            for s, nbrs in neigh.items():
                for k in range(0, len(nbrs), 4):
                    chunk = nbrs[k: k + 4]
                    line  = "CONECT" + f"{s:5d}" + "".join(f"{n:5d}" for n in chunk)
                    conect_lines.append(_ensure_nl80(line))

        out_path = f"{out_prefix}_pose{pose_idx:03d}.pdb"
        with open(out_path, "w") as fh:
            fh.writelines(
                rec_prefix + het_lines + conect_lines
                + (rec_terminal if rec_terminal else ["END\n"])
            )
        written.append(out_path)

    return written


# ════════════════════════════════════════════════════════════════════════════
#  STAGE-1 METADATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_stage1_metadata(mask_calc_dir: str) -> Dict[str, dict]:
    """Load Stage-1 .meta.json files. Returns {ligand_key → meta_dict}."""
    result: Dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(mask_calc_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
            lig = meta.get("ligand", {})
            key = (f"{lig.get('resname','?')}-"
                   f"{lig.get('chain','?')}-"
                   f"{lig.get('resseq','?')}")
            result[key] = meta
        except Exception as e:
            print(f"  ⚠️  Could not read {os.path.basename(path)}: {e}")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  MAIN DOCKING FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_stage6(
    pred_dir:       Optional[str] = None,
    mask_calc_dir:  Optional[str] = None,
    stage6_dir:     Optional[str] = None,
    gnina_bin:      Optional[str] = None,
    n_poses:        int   = 1,
    pool_choice:    str   = "both",
) -> Dict[str, dict]:
    """
    Run Stage-6 docking for all generated molecules.

    Returns
    -------
    Dict keyed by ligand_id:
        {
          "n_molecules" : int,
          "n_docked"    : int,
          "n_cached"    : int,   ← molecules skipped because already docked
          "n_failed"    : int,
          "csv_path"    : str,
        }
    """
    pred_dir      = pred_dir      or config.PRED_DIR
    mask_calc_dir = mask_calc_dir or config.MASK_CALC_OUTDIR
    stage6_dir    = stage6_dir    or config.STAGE6_DIR
    gnina_bin     = gnina_bin     or config.GNINA_BINARY

    os.makedirs(stage6_dir, exist_ok=True)

    gnina_bin   = ensure_gnina(gnina_bin, config.GNINA_DOWNLOAD_URL)
    stage1_meta = load_stage1_metadata(mask_calc_dir)
    print(f"  Stage-1 metadata loaded: {len(stage1_meta)} ligand(s)")

    lig_dirs = sorted([
        d for d in glob.glob(os.path.join(pred_dir, "*"))
        if os.path.isdir(d)
    ])
    if not lig_dirs:
        print(f"  ❌ No ligand sub-folders found in {pred_dir}")
        return {}
    print(f"  Ligand folders found: {len(lig_dirs)}")

    csv_path = os.path.join(stage6_dir, "docking_summary.csv")
    csv_rows: List[dict] = []
    all_results: Dict[str, dict] = {}

    for lig_dir in lig_dirs:
        lig_id = os.path.basename(lig_dir)

        print(f"\n{'='*60}")
        print(f"Ligand: {lig_id}")
        print(f"{'='*60}")

        meta = stage1_meta.get(lig_id)
        if meta is None:
            print(f"  ⚠️  No Stage-1 metadata found for '{lig_id}'. Skipping.")
            continue

        pdb_path = meta.get("pdb", "")
        lig_info = meta.get("ligand", {})
        resname  = lig_info.get("resname", "")
        chain    = lig_info.get("chain",   "")
        resseq   = int(lig_info.get("resseq", 1))

        if not os.path.exists(pdb_path):
            print(f"  ⚠️  PDB file not found: {pdb_path}. Skipping.")
            continue

        pool = load_pool_for_ligand(lig_dir, pool_choice)
        print(f"  Generated molecules to dock: {len(pool)}")
        if not pool:
            print("  ⚠️  Empty pool. Skipping.")
            continue

        lig_out_dir = os.path.join(stage6_dir, _safe(lig_id))
        os.makedirs(lig_out_dir, exist_ok=True)

        rec_pdb  = os.path.join(lig_out_dir, "rec.pdb")
        orig_pdb = os.path.join(lig_out_dir, "orig.pdb")

        if not prepare_receptor(pdb_path, rec_pdb):
            print(f"  ⚠️  Receptor preparation failed. Skipping {lig_id}.")
            continue
        if not prepare_autobox_ligand(pdb_path, orig_pdb, resname, chain, resseq):
            print(f"  ⚠️  Autobox ligand preparation failed. Skipping {lig_id}.")
            continue

        n_docked = 0
        n_cached = 0
        n_failed = 0

        for mol_idx, smiles in enumerate(pool, start=1):
            mol_dir  = os.path.join(lig_out_dir, f"mol_{mol_idx:04d}")
            os.makedirs(mol_dir, exist_ok=True)

            lig_sdf  = os.path.join(mol_dir, "ligand.sdf")
            out_sdf  = os.path.join(mol_dir, "docked_poses.sdf")
            log_file = os.path.join(mol_dir, "docking.log")
            prefix   = os.path.join(mol_dir, "complex")

            print(f"  [{mol_idx:>4d}/{len(pool)}] {smiles[:60]}", end=" ... ", flush=True)

            # ── Cache check (Assumption A11) ──────────────────────────────────
            # A molecule is already docked if its output SDF exists, is non-empty,
            # and RDKit can parse at least one valid pose from it.
            already_docked = (
                os.path.isfile(out_sdf)
                and os.path.getsize(out_sdf) > 0
                and len(parse_gnina_scores(out_sdf)) > 0
            )

            if already_docked:
                print("cached ✓", flush=True)
                scores = parse_gnina_scores(out_sdf)
                # Reconstruct which complex PDB files exist from the previous run
                written = sorted(
                    p for p in [
                        f"{prefix}_pose{pose_n:03d}.pdb"
                        for pose_n in range(1, len(scores) + 1)
                    ]
                    if os.path.isfile(p)
                )
                n_cached += 1

            else:
                # SMILES → SDF
                if not smiles_to_sdf(smiles, lig_sdf, name=f"mol_{mol_idx:04d}"):
                    print("3D embedding failed. Skipped.")
                    n_failed += 1
                    continue

                # Dock
                ok = run_gnina(
                    gnina_bin=gnina_bin, receptor=rec_pdb,
                    ligand=lig_sdf, autobox=orig_pdb,
                    out_sdf=out_sdf, log_file=log_file,
                    n_poses=9,   # always generate 9; filter on write
                )
                if not ok:
                    print("Docking failed. Skipped.")
                    n_failed += 1
                    continue

                scores = parse_gnina_scores(out_sdf)
                print(f"OK  ({len(scores)} poses)")

                written = write_complexes_from_gnina_sdf(
                    receptor_pdb=rec_pdb, poses_sdf=out_sdf,
                    out_prefix=prefix,
                    ligand_resname=resname, chain_id=chain, resseq=resseq,
                    write_conect=True, unique_atom_names=True,
                    max_poses=n_poses,
                )
                n_docked += 1

            # ── Append to CSV (both fresh and cached) ─────────────────────────
            for score_row in scores:
                pose_num = score_row["pose"]
                pdb_file = written[pose_num - 1] if pose_num <= len(written) else ""
                csv_rows.append({
                    "ligand_group":      lig_id,
                    "mol_idx":           mol_idx,
                    "smiles":            smiles,
                    "pose":              pose_num,
                    "CNNscore":          score_row.get("CNNscore", ""),
                    "CNNaffinity":       score_row.get("CNNaffinity", ""),
                    "minimizedAffinity": score_row.get("minimizedAffinity", ""),
                    "complex_pdb":       pdb_file,
                })

        all_results[lig_id] = {
            "n_molecules": len(pool),
            "n_docked":    n_docked,
            "n_cached":    n_cached,
            "n_failed":    n_failed,
            "csv_path":    csv_path,
        }
        print(f"\n  {lig_id}: {n_docked} newly docked, "
              f"{n_cached} from cache, {n_failed} failed.")

    # ── Write global summary CSV ──────────────────────────────────────────────
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_path, index=False)
        print(f"\n  💾 Docking summary CSV saved: {csv_path}")
        print(f"     {len(df)} rows  ({df['ligand_group'].nunique()} groups)")
    else:
        print("\n  ⚠️  No docking results to write.")

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

# ── Visualisation constants ───────────────────────────────────────────────────
_VIS_SCORE_COLS = ["CNNaffinity", "minimizedAffinity", "CNNscore"]

_VIS_SCORE_LABELS = {
    "CNNaffinity":       "CNN Affinity  (–log Kd/Ki)  ↑ higher = better",
    "minimizedAffinity": "Vinardo Score  (kcal mol⁻¹)  ↓ more negative = better",
    "CNNscore":          "CNN Score  (probability)  ↑ higher = better",
}

_VIS_LOWER_IS_BETTER = {"minimizedAffinity"}

_VIS_PANEL_COLORS = {
    "CNNaffinity":       "#2E86AB",   # steel blue
    "minimizedAffinity": "#E84855",   # crimson
    "CNNscore":          "#3BB273",   # green
}

_VIS_RANK_COLORS = {
    1: "#D4A017",   # gold
    2: "#8A8A8A",   # silver
    3: "#A0522D",   # bronze
}


def _draw_table(ax: plt.Axes, df_plot: pd.DataFrame) -> None:
    """Draw a styled summary table into ax (called by _make_barchart_figure)."""
    ax.axis("off")

    tbl_data = []
    for _, row in df_plot.iterrows():
        smi = textwrap.shorten(str(row["smiles"]), width=30, placeholder="…")
        tbl_data.append([
            str(int(row["rank"])),
            row["mol_label"],
            f"{row['CNNaffinity']:.3f}",
            f"{row['minimizedAffinity']:.3f}",
            f"{row['CNNscore']:.3f}",
            smi,
        ])

    headers = ["Rank", "Molecule", "CNNaff ↑", "Vinardo ↓", "CNNsc ↑", "SMILES"]
    col_w   = [0.06, 0.17, 0.11, 0.11, 0.09, 0.46]

    tbl = ax.table(cellText=tbl_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)

    for j in range(len(headers)):
        c = tbl[0, j]
        c.set_facecolor("#1A1A2E")
        c.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        c.set_edgecolor("#FAFAFA")
        c.set_width(col_w[j])

    hi = ["#D4A017", "#8A8A8A", "#A0522D"]   # gold / silver / bronze
    for ri in range(1, len(tbl_data) + 1):
        for ci in range(len(headers)):
            cell = tbl[ri, ci]
            if ri <= 3:
                cell.set_facecolor(hi[ri - 1] + "44")
                cell.set_text_props(fontweight="bold")
            elif ri % 2 == 0:
                cell.set_facecolor("#F0F4F8")
            else:
                cell.set_facecolor("#FFFFFF")
            cell.set_edgecolor("#E0E0E0")
            cell.set_width(col_w[ci])

    ax.set_title("Summary Table  (best pose per molecule)",
                 fontsize=9, fontweight="bold", pad=6, color="#1A1A2E")


def _make_barchart_figure(
    df_plot: pd.DataFrame,
    title_tag: str,
    out_path: str,
) -> None:
    """
    Build one complete figure:
      LEFT  — 3 horizontal bar chart panels (one per GNINA score)
      RIGHT — styled summary table
      BOTTOM — colour-key legend strip

    Bars are gold/silver/bronze for ranks 1/2/3; panel colour otherwise.
    Numeric values are printed at the end of every bar.
    Figure height auto-scales with the number of molecules.
    """
    n     = len(df_plot)
    row_h = 0.54 if n <= 10 else 0.38
    fig_h = max(7.5, n * row_h + 3.8)

    fig = plt.figure(figsize=(20, fig_h), facecolor="#FAFAFA")

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 0.001], hspace=0,
        left=0.01, right=0.99, top=0.93, bottom=0.09,
    )
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[0],
        width_ratios=[1, 1, 1, 1.65], wspace=0.10,
    )
    axes_bar = [fig.add_subplot(inner[0, i]) for i in range(3)]
    ax_table = fig.add_subplot(inner[0, 3])

    fig.suptitle(
        f"BRD4 Molecular Docking — Scores per Molecule  [{title_tag}]",
        fontsize=14, fontweight="bold", y=0.97, color="#1A1A2E",
    )

    y_pos    = np.arange(n)
    y_labels = df_plot["mol_label"].tolist()

    for i, col in enumerate(_VIS_SCORE_COLS):
        ax    = axes_bar[i]
        color = _VIS_PANEL_COLORS[col]
        vals  = df_plot[col].values

        bar_colors = [
            _VIS_RANK_COLORS.get(int(row["rank"]), color)
            for _, row in df_plot.iterrows()
        ]

        if col in _VIS_LOWER_IS_BETTER:
            ax.axvline(0, color="#999999", linewidth=0.8, alpha=0.5)

        ax.barh(y_pos, vals, color=bar_colors,
                edgecolor="white", linewidth=0.5,
                height=0.65, alpha=0.88)

        x_range = float(np.nanmax(np.abs(vals))) or 1.0
        offset  = x_range * 0.025
        for yi, v in zip(y_pos, vals):
            if not np.isfinite(v):
                continue
            ha   = "left"  if v >= 0 else "right"
            xpos = v + (offset if v >= 0 else -offset)
            ax.text(xpos, yi, f"{v:.2f}",
                    va="center", ha=ha, fontsize=6.8, color="#222222",
                    fontweight="bold" if yi < 3 else "normal")

        for yi in y_pos:
            ax.axhline(yi, color="#EBEBEB", linewidth=0.4, zorder=0)

        ax.set_yticks(y_pos)
        if i == 0:
            ax.set_yticklabels(y_labels, fontsize=7.5, color="#222222")
        else:
            ax.set_yticklabels([])

        ax.invert_yaxis()
        ax.set_xlabel(_VIS_SCORE_LABELS[col], fontsize=8, labelpad=5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#AAAAAA")
        ax.set_facecolor("#FAFAFA")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        direction = "▼ lower = better" if col in _VIS_LOWER_IS_BETTER else "▲ higher = better"
        ax.set_title(f"{col}\n{direction}",
                     fontsize=9, fontweight="bold", color=color, pad=4)

    axes_bar[0].set_ylabel("Molecule  (ranked by CNN affinity)",
                            fontsize=9, labelpad=8, color="#555")

    _draw_table(ax_table, df_plot)

    ax_leg = fig.add_axes([0.13, 0.01, 0.55, 0.055])
    ax_leg.axis("off")
    legend_patches = [
        mpatches.Patch(color=_VIS_RANK_COLORS[1], label="Rank 1  (best)"),
        mpatches.Patch(color=_VIS_RANK_COLORS[2], label="Rank 2"),
        mpatches.Patch(color=_VIS_RANK_COLORS[3], label="Rank 3"),
        mpatches.Patch(color="#2E86AB", alpha=0.7, label="Other — CNNaffinity panel"),
        mpatches.Patch(color="#E84855", alpha=0.7, label="Other — Vinardo panel"),
        mpatches.Patch(color="#3BB273", alpha=0.7, label="Other — CNNscore panel"),
    ]
    ax_leg.legend(
        handles=legend_patches,
        loc="center left", ncol=6, fontsize=7.5, frameon=True,
        framealpha=0.85, edgecolor="#CCCCCC",
        title="Bar colour key", title_fontsize=7.5,
    )

    fig.text(
        0.72, 0.025,
        "Ranking: CNNaffinity (primary) + Vinardo (tiebreak)\n"
        "Best pose (pose 1) per molecule | GNINA v1.0.3",
        fontsize=6.5, color="#888888", style="italic", ha="left", va="center",
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"  🖼  Saved: {out_path}")


def run_visualisation(
    csv_path:  Optional[str] = None,
    out_dir:   Optional[str] = None,
) -> None:
    """
    Read docking_summary.csv and produce two ranking bar-chart PNGs:
      1. Top-10 molecules  → stage6_docking_top10_molecules_all_scores_BARCHART.png
      2. All molecules     → stage6_docking_all_molecules_all_scores_BARCHART.png

    Ranking: pose-1 CNNaffinity (primary, higher = better)
             + minimizedAffinity tiebreaker (lower = better, Assumption A12).

    Parameters
    ----------
    csv_path : path to docking_summary.csv  (default: config.STAGE6_DIR/docking_summary.csv)
    out_dir  : directory for PNG output     (default: config.STAGE6_DIR)
    """
    warnings.filterwarnings("ignore")

    csv_path = csv_path or os.path.join(config.STAGE6_DIR, "docking_summary.csv")
    out_dir  = out_dir  or config.STAGE6_DIR

    if not os.path.exists(csv_path):
        print(f"  ⚠️  docking_summary.csv not found at {csv_path}. "
              "Skipping visualisation.")
        return

    df_raw = pd.read_csv(csv_path)

    # Coerce score columns (GNINA writes empty string when a property is missing)
    for c in _VIS_SCORE_COLS:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    # Keep best pose only (pose 1 = GNINA top-ranked pose)
    df = df_raw[df_raw["pose"] == 1].copy().reset_index(drop=True)

    if df.empty:
        print("  ⚠️  No pose-1 rows found in CSV. Skipping visualisation.")
        return

    # Rank: CNNaffinity primary (higher = better),
    #       minimizedAffinity tiebreaker (lower = better → subtract)
    df["_rank_key"] = df["CNNaffinity"] - 0.01 * df["minimizedAffinity"]
    df = df.sort_values("_rank_key", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Clean y-axis label:  "EAM-A-1  mol-3"
    df["mol_label"] = (
        df["ligand_group"].str.replace("_", " ", regex=False)
        + "  mol-" + df["mol_idx"].astype(str)
    )

    print(f"\n  Visualisation — {len(df)} molecule(s) across "
          f"{df['ligand_group'].nunique()} group(s)")

    # Figure 1: top 10
    _make_barchart_figure(
        df.head(10).copy(),
        title_tag="Top 10 Molecules",
        out_path=os.path.join(
            out_dir,
            "stage6_docking_top10_molecules_all_scores_BARCHART.png",
        ),
    )

    # Figure 2: all molecules
    _make_barchart_figure(
        df.copy(),
        title_tag=f"All {len(df)} Molecules",
        out_path=os.path.join(
            out_dir,
            "stage6_docking_all_molecules_all_scores_BARCHART.png",
        ),
    )

    print("  ✅ Visualisation complete.")


# ════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED TEST  (no real GNINA or GPU required)
# ════════════════════════════════════════════════════════════════════════════

_MOCK_SDF = """\

     RDKit          3D

  6  6  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1000    1.2124    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4000    2.4248    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    2.4248    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7000    1.2124    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END
> <CNNscore>
0.92

> <CNNaffinity>
8.5

> <minimizedAffinity>
-9.1

$$$$

     RDKit          3D

  6  6  0  0  0  0  0  0  0  0999 V2000
    0.1000    0.1000    0.1000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.1000    0.1000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.2000    1.3124    0.1000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    2.5248    0.1000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.1000    2.5248    0.1000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6000    1.3124    0.1000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END
> <CNNscore>
0.75

> <CNNaffinity>
7.2

> <minimizedAffinity>
-7.8

$$$$
"""

_MOCK_REC_PDB = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   1.000   1.000  1.00  0.00           C
END
"""


def _run_test() -> bool:
    import stat as stat_mod

    print("\n" + "=" * 60)
    print("STAGE 6 SELF-TEST  (mock GNINA, no GPU required)")
    print("=" * 60)

    td = os.path.join(config.STAGE6_DIR, "test")
    if os.path.exists(td):
        shutil.rmtree(td)
        print(f"  🗑  Wiped existing test dir: {td}")
    os.makedirs(td)
    print(f"  📁 Created fresh test dir:  {td}\n")

    pred_dir      = os.path.join(td, "preds")
    mask_calc_dir = os.path.join(td, "stage1")
    stage6_dir    = os.path.join(td, "stage6_out")
    pdb_dir       = os.path.join(td, "pdb")

    os.makedirs(pdb_dir)
    pdb_path = os.path.join(pdb_dir, "3MXF.pdb")
    with open(pdb_path, "w") as f:
        f.write(_MOCK_REC_PDB.replace("END\n", ""))
        f.write("HETATM    4  C1  JQ1 A   1       5.000   5.000   5.000"
                "  1.00  0.00           C\n")
        f.write("END\n")

    os.makedirs(mask_calc_dir)
    lig_id  = "JQ1-A-1"
    s1_meta = {
        "smiles":              "c1ccccc1",
        "masked_atom_indices": [0, 1],
        "ligand":              {"resname": "JQ1", "chain": "A", "resseq": 1},
        "pdb":                 pdb_path,
    }
    with open(os.path.join(mask_calc_dir, "JQ1-A-1.meta.json"), "w") as f:
        json.dump(s1_meta, f)

    lig_dir = os.path.join(pred_dir, _safe(lig_id))
    os.makedirs(lig_dir)
    test_smiles = ["c1ccccc1", "CC(=O)O"]
    for mc in range(1, 3):
        with open(os.path.join(lig_dir, f"ia_mask{mc:03d}.txt"), "w") as f:
            f.write("\n".join(test_smiles) + "\n")
        with open(os.path.join(lig_dir, f"rand_mask{mc:03d}.txt"), "w") as f:
            f.write(test_smiles[0] + "\n")

    mock_sdf_path = os.path.join(td, "mock_poses.sdf")
    with open(mock_sdf_path, "w") as f:
        f.write(_MOCK_SDF)

    mock_gnina = os.path.join(td, "mock_gnina")
    mock_script = (
        "#!/bin/bash\n"
        "OUT=''\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ \"$1\" == '--out' ]]; then OUT=\"$2\"; shift 2\n"
        "  else shift; fi\n"
        "done\n"
        f"cp '{mock_sdf_path}' \"$OUT\"\n"
        "exit 0\n"
    )
    with open(mock_gnina, "w") as f:
        f.write(mock_script)
    os.chmod(mock_gnina,
             os.stat(mock_gnina).st_mode
             | stat_mod.S_IXUSR | stat_mod.S_IXGRP | stat_mod.S_IXOTH)

    passed = 0
    failed = 0

    def check(condition: bool, msg: str):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ PASS  {msg}")
            passed += 1
        else:
            print(f"  ❌ FAIL  {msg}")
            failed += 1

    # ── First run — docks everything fresh ───────────────────────────────────
    results = run_stage6(
        pred_dir=pred_dir, mask_calc_dir=mask_calc_dir,
        stage6_dir=stage6_dir, gnina_bin=mock_gnina,
        n_poses=2, pool_choice="both",
    )

    check(lig_id in results, f"'{lig_id}' present in results")

    if lig_id in results:
        r = results[lig_id]
        check(r["n_molecules"] > 0,       f"pool non-empty (n={r['n_molecules']})")
        check(r["n_docked"]    > 0,       f"at least one molecule docked (n={r['n_docked']})")
        check(r["n_cached"]    == 0,      "n_cached == 0 on first run")
        check(r["n_failed"]    == 0,      "n_failed == 0")
        check(os.path.exists(r["csv_path"]), "docking_summary.csv on disk")

        df = pd.read_csv(r["csv_path"])
        check(len(df) > 0, f"CSV has rows (got {len(df)})")
        check({"CNNscore", "CNNaffinity", "minimizedAffinity"}.issubset(df.columns),
              "CSV has all three score columns")

        lig_out  = os.path.join(stage6_dir, _safe(lig_id))
        pdb_files = glob.glob(os.path.join(lig_out, "**", "complex_pose*.pdb"),
                               recursive=True)
        check(len(pdb_files) > 0, f"complex PDB files written (found {len(pdb_files)})")

        per_mol = (df[df["complex_pdb"] != ""]
                   .groupby("mol_idx")["complex_pdb"].nunique())
        if len(per_mol) > 0:
            check(per_mol.max() <= 2, f"≤ 2 complex PDBs per molecule (max={per_mol.max()})")

    # ── Second run — everything should be served from cache ───────────────────
    results2 = run_stage6(
        pred_dir=pred_dir, mask_calc_dir=mask_calc_dir,
        stage6_dir=stage6_dir, gnina_bin=mock_gnina,
        n_poses=2, pool_choice="both",
    )

    if lig_id in results2:
        r2 = results2[lig_id]
        check(r2["n_docked"] == 0,
              f"n_docked == 0 on second run (got {r2['n_docked']})")
        check(r2["n_cached"] == r2["n_molecules"],
              f"all {r2['n_molecules']} molecules served from cache")

    # ── Visualisation smoke test ──────────────────────────────────────────────
    if lig_id in results and os.path.exists(results[lig_id]["csv_path"]):
        vis_dir = os.path.join(td, "vis_out")
        run_visualisation(
            csv_path=results[lig_id]["csv_path"],
            out_dir=vis_dir,
        )
        top10_png = os.path.join(
            vis_dir,
            "stage6_docking_top10_molecules_all_scores_BARCHART.png",
        )
        check(os.path.exists(top10_png), "top-10 barchart PNG produced")

    print(f"\n{'='*60}")
    print(f"Test complete:  {passed} passed,  {failed} failed.")
    print("✅ All tests passed." if failed == 0 else "❌ Some tests failed.")
    print(f"\n  Test outputs preserved at: {td}")
    return failed == 0


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("STAGE 6: MOLECULAR DOCKING WITH GNINA")
    print("=" * 60)

    print(f"""
  This stage docks all generated molecules (from Stage 2) into the BRD4
  binding pocket using GNINA with autobox, then produces ranking plots.

  GNINA binary : {config.GNINA_BINARY}
  (Auto-downloaded from GitHub v1.0.3 if not found.)

  ⚠️  GNINA is Linux/x86_64 only. On macOS / Windows use WSL.
  ⚠️  GPU (CUDA) is strongly recommended; CPU docking is very slow.
  ⚠️  Already-docked molecules are skipped automatically (cached ✓).

  Before running the full pipeline you can run a smoke test instead.
  The smoke test uses a mock GNINA script — no binary, no GPU needed.
  Writes to: {os.path.join(config.STAGE6_DIR, "test")}
  (wiped clean at the start of every test run).
""")

    run_test = _ask_yes_no_default("Run the smoke test?", default=False)
    if run_test:
        ok = _run_test()
        sys.exit(0 if ok else 1)

    n_poses = ask_runtime_options()

    # ── Generator stage ───────────────────────────────────────────────────────
    gen_stage = _ask_generator_stage()
    pred_dir  = _resolve_pred_dir(gen_stage)

    # ── Pool choice ───────────────────────────────────────────────────────────
    print("\n  Pool choice for docking:")
    print("    ia   : interaction-aware only")
    print("    rand : random only")
    print("    both : IA + random, deduplicated  [default]")
    while True:
        raw = input("  Pool choice (ia / rand / both) [both]: ").strip().lower()
        if raw == "":
            pool = "both"; break
        if raw in ("ia", "rand", "both"):
            pool = raw; break
        print("    Please type 'ia', 'rand', or 'both'.")

    # ── Nested output directory (A1: CSV lives inside this dir) ───────────────
    nested_dir = _nested_stage6_dir(gen_stage, pool)
    csv_path   = os.path.join(nested_dir, "docking_summary.csv")

    print(f"\n  Generator stage : Stage {gen_stage}")
    print(f"  pred_dir        : {pred_dir}")
    print(f"  Output dir      : {nested_dir}")
    print(f"  CSV             : {csv_path}\n")

    results = run_stage6(
        pred_dir    = pred_dir,
        stage6_dir  = nested_dir,
        n_poses     = n_poses,
        pool_choice = pool,
    )

    print("\n" + "=" * 60)
    print("✅ Docking complete.")
    print(f"   Generator stage  : Stage {gen_stage}")
    print(f"   Pool             : {pool}")
    print(f"   Outputs saved to : {nested_dir}")
    print(f"   Summary CSV      : {csv_path}")
    print(f"   Ligands processed: {len(results)}")
    for lig_id, r in sorted(results.items()):
        print(f"   • {lig_id}  docked={r['n_docked']}  "
              f"cached={r['n_cached']}  failed={r['n_failed']}")

    # ── Visualisation — nested CSV and output dir ─────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 6 VISUALISATION")
    print("=" * 60)
    run_visualisation(csv_path=csv_path, out_dir=nested_dir)

    print("\n" + "=" * 60)
    print("✅ Stage 6 complete.")
    print(f"   Plots saved to: {nested_dir}")

if __name__ == "__main__":
    main()

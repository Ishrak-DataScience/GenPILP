# -*- coding: utf-8 -*-
"""
stage6_docking.py
=================
Stage 6 of the pipeline — molecular docking of generated molecules.

Uses GNINA (v1.0.3) with autobox to dock all generated molecules from
Stage 2 into the BRD4 binding pocket defined by the original ligand
in each PDB structure.

Per ligand group (e.g. EAM-A-1), produces:
  docked/<group>/<molecule_idx>/
      ligand.sdf                    — 3D conformer input to GNINA
      docked_poses.sdf              — GNINA multi-pose output
      complex_pose001.pdb           — receptor + best pose (or all N poses)
      complex_pose002.pdb           — (if n_poses > 1)
      ...
  docking_summary.csv               — one row per (group, molecule, pose)
      columns: ligand_group, mol_idx, smiles, pose, CNNscore,
               CNNaffinity, minimizedAffinity, complex_pdb

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A1.  GNINA is Linux-only (x86_64). The auto-download fetches the
     pre-compiled binary from GitHub. This works in Colab and on any
     Linux machine. macOS / Windows require WSL or manual compilation.

A2.  GNINA binary location is config.GNINA_BINARY (default "./gnina").
     If not found, Stage 6 auto-downloads it (Ambiguity 2 Option A).
     The download URL is config.GNINA_DOWNLOAD_URL.

A3.  Receptor preparation: ATOM records from the PDB are written to
     rec.pdb (matching the notebook Cell 9). HETATM records matching
     resname + chain + resseq are written to orig.pdb (autobox ligand).

A4.  SMILES → 3D SDF conversion uses RDKit ETKDG + UFF minimisation
     (identical to notebook Cell 7). If 3D embedding fails for a
     molecule it is skipped with a warning.

A5.  The PDB file path and ligand identity (resname, chain, resseq) for
     each group are read from Stage-1 .meta.json files in
     config.MASK_CALC_OUTDIR, matched by the ligand key
     "{resname}-{chain}-{resseq}".

A6.  All generated molecules are docked (Ambiguity 1 Option A).

A7.  Number of poses to keep is asked at runtime (default 1).
     GNINA always generates up to 9 internally; this setting controls
     how many complex PDB files are written. Score CSV always contains
     all GNINA poses regardless of this setting.

A8.  Score properties read from GNINA's output SDF:
       CNNscore       — CNN-predicted binding probability  (higher = better)
       CNNaffinity    — CNN-predicted -log Kd/Ki            (higher = better)
       minimizedAffinity — Vinardo score (kcal/mol)          (more negative = better)
     If a property is absent in a pose, it is written as "" in the CSV.

A9.  The complex PDB writing code is transcribed verbatim from the
     reference notebook (Cell 12) to avoid any import dependency.

A10. The self-test creates a mock GNINA shell script that copies a
     pre-built minimal SDF to the expected output path so no real
     binary or GPU is required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN  (Colab)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Step 1 — run after stage2_molecule_generation.py
  # GNINA is downloaded automatically on first run (~500 MB)
  python stage6_docking.py

HOW TO RUN  (local Linux)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Optionally pre-download GNINA and set config.GNINA_BINARY to its path:
  #   wget https://github.com/gnina/gnina/releases/download/v1.0.3/gnina
  #   chmod +x gnina
  python stage6_docking.py

HOW TO TEST  (no GNINA, no GPU required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Answer "yes" to the smoke-test prompt.
  Writes to config.STAGE6_DIR/test/ (wiped each run).
"""

import csv
import glob
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

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
    # Guard: a previous failed run may have left a directory at dest_path
    # (e.g. if os.makedirs was mistakenly called with dest_path itself).
    # Remove it so urlretrieve can create a file at that path.
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
    # Make executable (equivalent to chmod +x)
    current = os.stat(dest_path).st_mode
    os.chmod(dest_path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  ✅ GNINA downloaded and made executable: {dest_path}")


# Local execution path — FUSE-mounted filesystems (Google Drive in Colab)
# do not support direct binary execution via execve().  We always copy the
# binary to a local path before running it.
_GNINA_LOCAL = "/content/gnina"


def _make_executable(path: str) -> None:
    """Set executable bits on a file (equivalent to chmod +x)."""
    current = os.stat(path).st_mode
    os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def ensure_gnina(
    binary_path: str = config.GNINA_BINARY,
    download_url: str = config.GNINA_DOWNLOAD_URL,
) -> str:
    """
    Return the path to a locally executable GNINA binary.

    Strategy (handles the Colab/Drive FUSE PermissionError):
      1. If config.GNINA_BINARY exists on Drive → copy to _GNINA_LOCAL
         and execute from there.
      2. If config.GNINA_BINARY is missing → download it to Drive first
         (permanent storage), then copy to _GNINA_LOCAL for execution.

    The Drive copy is the permanent store; _GNINA_LOCAL is the
    session-local executable copy.  Both are kept in sync.
    """
    drive_path = binary_path   # permanent storage (may be on Drive/FUSE)
    local_path = _GNINA_LOCAL  # session-local, always on a real filesystem

    # ── Step 1: ensure Drive copy exists ─────────────────────────────────────
    if not os.path.isfile(drive_path):
        print(f"  GNINA not found at '{drive_path}'. Downloading to Drive ...")
        _download_gnina(drive_path, download_url)
    else:
        print(f"  ✅ GNINA found on Drive: {drive_path}")

    # ── Step 2: copy to local filesystem if needed ────────────────────────────
    # Always copy when the local copy is missing or older than the Drive copy.
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
    """Ask how many GNINA poses to write as complex PDB files. Default 1."""
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
    Convert SMILES to a 3D SDF file using RDKit ETKDG + UFF.
    Returns True on success, False if 3D embedding fails (Assumption A4).
    Transcribed from notebook Cell 7.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"    ⚠️  Cannot parse SMILES: {smiles[:60]}")
        return False

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result == -1:
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
    """
    Write ATOM records only from pdb_path to rec_out (receptor without ligand).
    Mirrors notebook Cell 9: grep ATOM $Protein_File > rec.pdb
    Returns True if any ATOM lines were written.
    """
    os.makedirs(os.path.dirname(rec_out) or ".", exist_ok=True)
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("ATOM "):
                lines.append(line)
    if not lines:
        print(f"    ⚠️  No ATOM records found in {pdb_path}")
        return False
    with open(rec_out, "w") as f:
        f.writelines(lines)
        f.write("END\n")
    return True


def prepare_autobox_ligand(
    pdb_path: str,
    orig_out: str,
    resname: str,
    chain: str,
    resseq: int,
) -> bool:
    """
    Write HETATM records matching resname + chain + resseq to orig_out.
    This defines the autobox for GNINA docking.
    Mirrors notebook Cell 9: grep HETATM $Protein_File | grep "$lig_grep" > orig.pdb
    Returns True if any matching lines were written.
    """
    os.makedirs(os.path.dirname(orig_out) or ".", exist_ok=True)
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            line_res   = line[17:20].strip()
            line_chain = line[21].strip()
            try:
                line_seq = int(line[22:26])
            except ValueError:
                continue
            if (line_res == resname
                    and line_chain == chain
                    and line_seq == resseq):
                lines.append(line)
    if not lines:
        print(f"    ⚠️  No HETATM records found for "
              f"{resname}:{chain}:{resseq} in {pdb_path}")
        return False
    with open(orig_out, "w") as f:
        f.writelines(lines)
        f.write("END\n")
    return True


# ════════════════════════════════════════════════════════════════════════════
#  DOCKING
# ════════════════════════════════════════════════════════════════════════════

def run_gnina(
    gnina_bin: str,
    receptor:  str,
    ligand:    str,
    autobox:   str,
    out_sdf:   str,
    log_file:  str,
    n_poses:   int = 9,
) -> bool:
    """
    Run GNINA docking via subprocess.
    Returns True on success (exit code 0).
    Mirrors notebook Cell 10.
    """
    os.makedirs(os.path.dirname(out_sdf) or ".", exist_ok=True)
    cmd = [
        gnina_bin,
        "-r", receptor,
        "-l", ligand,
        "--autobox_ligand", autobox,
        "--out",  out_sdf,
        "--log",  log_file,
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


def parse_gnina_scores(
    sdf_path: str,
) -> List[Dict[str, str]]:
    """
    Read GNINA output SDF and extract per-pose scores (Assumption A8).
    Returns list of dicts, one per pose, keys: pose, CNNscore,
    CNNaffinity, minimizedAffinity.
    """
    if not os.path.exists(sdf_path):
        return []
    rows = []
    suppl = Chem.ForwardSDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    pose = 0
    for mol in suppl:
        if mol is None:
            continue
        pose += 1
        row = {"pose": pose}
        for prop in _SCORE_PROPS:
            row[prop] = mol.GetPropsAsDict().get(prop, "")
        rows.append(row)
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  COMPLEX WRITING — verbatim from notebook Cell 12 (Assumption A9)
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


def _split_before_terminal(
    records: List[str],
) -> Tuple[List[str], List[str]]:
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
    if len(e) == 1:
        return f"{n:>4}"[:4]
    else:
        return f"{n:<4}"[:4]


def _format_hetatm(
    serial: int, elem: str, atom_name: str,
    resname: str, chain: str, resseq: int,
    x: float, y: float, z: float,
    occ: float = 1.00, bfac: float = 0.00,
    alt: str = " ", icode: str = " ",
) -> str:
    e = (elem or "").strip().upper()
    if not e:
        e = _infer_element_from_name(atom_name)
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
                m,
                charge=total_charge,
                allowChargedFragments=True,
                embedChiral=True,
                useAtomMap=False,
            )
            Chem.SanitizeMol(m)
            yield m
        except Exception:
            yield m


def write_complexes_from_gnina_sdf(
    receptor_pdb:     str,
    poses_sdf:        str,
    out_prefix:       str,
    ligand_resname:   str = "LIG",
    chain_id:         str = "A",
    resseq:           int = 1,
    write_conect:     bool = True,
    unique_atom_names: bool = True,
    max_poses:        int = 9,
) -> List[str]:
    """
    Create one complex PDB per pose: receptor + ligand HETATM (+ CONECT).
    Returns list of written file paths.
    max_poses caps the number of complex PDBs written (Assumption A7).
    Transcribed verbatim from notebook Cell 12 (Assumption A9).
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
        het_lines: List[str] = []
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
                rec_prefix
                + het_lines
                + conect_lines
                + (rec_terminal if rec_terminal else ["END\n"])
            )
        written.append(out_path)

    return written


# ════════════════════════════════════════════════════════════════════════════
#  STAGE-1 METADATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_stage1_metadata(mask_calc_dir: str) -> Dict[str, dict]:
    """
    Load Stage-1 .meta.json files.
    Returns {ligand_key → meta_dict} where
    ligand_key = "{resname}-{chain}-{resseq}".
    """
    import json
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

    Parameters
    ----------
    pred_dir      : Stage-2 predictions folder  (default: config.PRED_DIR)
    mask_calc_dir : Stage-1 JSON folder          (default: config.MASK_CALC_OUTDIR)
    stage6_dir    : output root                  (default: config.STAGE6_DIR)
    gnina_bin     : GNINA binary path            (default: config.GNINA_BINARY)
    n_poses       : complex PDB files to write per molecule
    pool_choice   : 'ia', 'rand', or 'both'

    Returns
    -------
    Dict keyed by ligand_id:
        {
          "n_molecules" : int,
          "n_docked"    : int,
          "n_failed"    : int,
          "csv_path"    : str,
        }
    """
    pred_dir      = pred_dir      or config.PRED_DIR
    mask_calc_dir = mask_calc_dir or config.MASK_CALC_OUTDIR
    stage6_dir    = stage6_dir    or config.STAGE6_DIR
    gnina_bin     = gnina_bin     or config.GNINA_BINARY

    os.makedirs(stage6_dir, exist_ok=True)

    # ── Ensure GNINA is available ─────────────────────────────────────────────
    gnina_bin = ensure_gnina(gnina_bin, config.GNINA_DOWNLOAD_URL)

    # ── Load Stage-1 metadata (PDB paths + ligand identities) ─────────────────
    stage1_meta = load_stage1_metadata(mask_calc_dir)
    print(f"  Stage-1 metadata loaded: {len(stage1_meta)} ligand(s)")

    # ── Discover ligand prediction folders ───────────────────────────────────
    lig_dirs = sorted([
        d for d in glob.glob(os.path.join(pred_dir, "*"))
        if os.path.isdir(d)
    ])
    if not lig_dirs:
        print(f"  ❌ No ligand sub-folders found in {pred_dir}")
        return {}
    print(f"  Ligand folders found: {len(lig_dirs)}")

    # ── Global summary CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(stage6_dir, "docking_summary.csv")
    csv_rows: List[dict] = []

    all_results: Dict[str, dict] = {}

    for lig_dir in lig_dirs:
        lig_id = os.path.basename(lig_dir)

        print(f"\n{'='*60}")
        print(f"Ligand: {lig_id}")
        print(f"{'='*60}")

        # ── Match to Stage-1 metadata ─────────────────────────────────────────
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

        # ── Load generated molecules ──────────────────────────────────────────
        pool = load_pool_for_ligand(lig_dir, pool_choice)
        print(f"  Generated molecules to dock: {len(pool)}")
        if not pool:
            print("  ⚠️  Empty pool. Skipping.")
            continue

        lig_out_dir = os.path.join(stage6_dir, _safe(lig_id))
        os.makedirs(lig_out_dir, exist_ok=True)

        # ── Prepare receptor and autobox once per ligand group ─────────────────
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

            # ── Cache check: skip docking if a valid output SDF already exists ──
            # A non-empty docked_poses.sdf means this molecule was already docked
            # successfully in a previous run.  Re-parse its scores and reuse them
            # without calling GNINA again.
            already_docked = (
                os.path.isfile(out_sdf)
                and os.path.getsize(out_sdf) > 0
                and len(parse_gnina_scores(out_sdf)) > 0
            )

            if already_docked:
                print("cached ✓", flush=True)
                scores = parse_gnina_scores(out_sdf)
                # Reconstruct the list of complex PDB paths that were written
                # previously so the CSV column stays accurate.
                written = sorted(
                    f for f in [
                        f"{prefix}_pose{p:03d}.pdb"
                        for p in range(1, len(scores) + 1)
                    ]
                    if os.path.isfile(f)
                )
                n_cached += 1

            else:
                # ── SMILES → SDF ──────────────────────────────────────────────
                if not smiles_to_sdf(smiles, lig_sdf, name=f"mol_{mol_idx:04d}"):
                    print("3D embedding failed. Skipped.")
                    n_failed += 1
                    continue

                # ── Dock ──────────────────────────────────────────────────────
                ok = run_gnina(
                    gnina_bin = gnina_bin,
                    receptor  = rec_pdb,
                    ligand    = lig_sdf,
                    autobox   = orig_pdb,
                    out_sdf   = out_sdf,
                    log_file  = log_file,
                    n_poses   = 9,    # always generate 9; filter on write
                )
                if not ok:
                    print("Docking failed. Skipped.")
                    n_failed += 1
                    continue

                # ── Parse scores ──────────────────────────────────────────────
                scores = parse_gnina_scores(out_sdf)
                print(f"OK  ({len(scores)} poses)")

                # ── Write complex PDB files (up to n_poses) ───────────────────
                written = write_complexes_from_gnina_sdf(
                    receptor_pdb   = rec_pdb,
                    poses_sdf      = out_sdf,
                    out_prefix     = prefix,
                    ligand_resname = resname,
                    chain_id       = chain,
                    resseq         = resseq,
                    write_conect   = True,
                    unique_atom_names = True,
                    max_poses      = n_poses,
                )
                n_docked += 1

            # ── Append to summary CSV (both fresh and cached) ─────────────────
            for score_row in scores:
                pose_num = score_row["pose"]
                pdb_file = written[pose_num - 1] if pose_num <= len(written) else ""
                csv_rows.append({
                    "ligand_group":       lig_id,
                    "mol_idx":            mol_idx,
                    "smiles":             smiles,
                    "pose":               pose_num,
                    "CNNscore":           score_row.get("CNNscore", ""),
                    "CNNaffinity":        score_row.get("CNNaffinity", ""),
                    "minimizedAffinity":  score_row.get("minimizedAffinity", ""),
                    "complex_pdb":        pdb_file,
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
#  SELF-CONTAINED TEST  (no real GNINA or GPU required)
# ════════════════════════════════════════════════════════════════════════════

# Minimal valid multi-pose SDF with two poses and fake GNINA score properties
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

# Minimal receptor PDB (just 3 backbone atoms)
_MOCK_REC_PDB = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   1.000   1.000  1.00  0.00           C
END
"""

# Minimal original ligand PDB (benzene as autobox reference)
_MOCK_ORIG_PDB = """\
HETATM    1  C1  BNZ A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  BNZ A   1       1.400   0.000   0.000  1.00  0.00           C
END
"""


def _run_test() -> bool:
    print("\n" + "=" * 60)
    print("STAGE 6 SELF-TEST  (mock GNINA, no GPU required)")
    print("=" * 60)

    td = os.path.join(config.STAGE6_DIR, "test")
    if os.path.exists(td):
        shutil.rmtree(td)
        print(f"  🗑  Wiped existing test dir: {td}")
    os.makedirs(td)
    print(f"  📁 Created fresh test dir:  {td}\n")

    import json, stat as stat_mod

    pred_dir      = os.path.join(td, "preds")
    mask_calc_dir = os.path.join(td, "stage1")
    stage6_dir    = os.path.join(td, "stage6_out")
    pdb_dir       = os.path.join(td, "pdb")

    # ── Write a minimal PDB file ──────────────────────────────────────────────
    os.makedirs(pdb_dir)
    pdb_path = os.path.join(pdb_dir, "3MXF.pdb")
    with open(pdb_path, "w") as f:
        f.write(_MOCK_REC_PDB.replace("END\n", ""))
        # Add an HETATM for the autobox ligand
        f.write("HETATM    4  C1  JQ1 A   1       5.000   5.000   5.000"
                "  1.00  0.00           C\n")
        f.write("END\n")

    # ── Write Stage-1 JSON ────────────────────────────────────────────────────
    os.makedirs(mask_calc_dir)
    lig_id   = "JQ1-A-1"
    s1_meta  = {
        "smiles":              "c1ccccc1",
        "masked_atom_indices": [0, 1],
        "ligand":              {"resname": "JQ1", "chain": "A", "resseq": 1},
        "pdb":                 pdb_path,
    }
    with open(os.path.join(mask_calc_dir, "JQ1-A-1.meta.json"), "w") as f:
        json.dump(s1_meta, f)

    # ── Write Stage-2 prediction txt files ───────────────────────────────────
    lig_dir = os.path.join(pred_dir, _safe(lig_id))
    os.makedirs(lig_dir)
    test_smiles = ["c1ccccc1", "CC(=O)O"]   # benzene, acetic acid
    for mc in range(1, 3):
        with open(os.path.join(lig_dir, f"ia_mask{mc:03d}.txt"), "w") as f:
            f.write("\n".join(test_smiles) + "\n")
        with open(os.path.join(lig_dir, f"rand_mask{mc:03d}.txt"), "w") as f:
            f.write(test_smiles[0] + "\n")

    # ── Create mock GNINA binary (shell script that copies mock SDF) ──────────
    mock_sdf_path = os.path.join(td, "mock_poses.sdf")
    with open(mock_sdf_path, "w") as f:
        f.write(_MOCK_SDF)

    mock_gnina = os.path.join(td, "mock_gnina")
    # The script reads --out from its arguments and copies the mock SDF there
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

    # ── Run Stage 6 ───────────────────────────────────────────────────────────
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

    results = run_stage6(
        pred_dir      = pred_dir,
        mask_calc_dir = mask_calc_dir,
        stage6_dir    = stage6_dir,
        gnina_bin     = mock_gnina,
        n_poses       = 2,
        pool_choice   = "both",
    )

    check(lig_id in results,
          f"'{lig_id}' present in results")

    if lig_id in results:
        r = results[lig_id]
        check(r["n_molecules"] > 0,
              f"pool non-empty (n={r['n_molecules']})")
        check(r["n_docked"] > 0,
              f"at least one molecule docked (n={r['n_docked']})")
        check(os.path.exists(r["csv_path"]),
              "docking_summary.csv on disk")

        # Check CSV content
        df = pd.read_csv(r["csv_path"])
        check(len(df) > 0,
              f"CSV has rows (got {len(df)})")
        check(set(["CNNscore", "CNNaffinity", "minimizedAffinity"]).issubset(df.columns),
              "CSV has all three score columns")
        check("complex_pdb" in df.columns,
              "CSV has complex_pdb column")

        # Check that complex PDB files were written
        lig_out = os.path.join(stage6_dir, _safe(lig_id))
        pdb_files = glob.glob(os.path.join(lig_out, "**", "complex_pose*.pdb"),
                              recursive=True)
        check(len(pdb_files) > 0,
              f"at least one complex PDB written (found {len(pdb_files)})")

        # Scores should be floats
        sample_score = df["CNNscore"].dropna()
        if len(sample_score) > 0:
            try:
                float(sample_score.iloc[0])
                check(True, "CNNscore parses as float")
            except ValueError:
                check(False, "CNNscore parses as float")

        # n_poses=2 → at most 2 complex PDBs per molecule
        per_mol_counts = (
            df[df["complex_pdb"] != ""]
            .groupby("mol_idx")["complex_pdb"]
            .nunique()
        )
        if len(per_mol_counts) > 0:
            check(per_mol_counts.max() <= 2,
                  f"at most 2 complex PDBs per molecule (max={per_mol_counts.max()})")

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
  binding pocket using GNINA with autobox.

  GNINA binary : {config.GNINA_BINARY}
  (Auto-downloaded from GitHub v1.0.3 if not found.)

  ⚠️  GNINA is Linux/x86_64 only. On macOS / Windows use WSL.
  ⚠️  GPU (CUDA) is strongly recommended; CPU docking is very slow.

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

    results = run_stage6(n_poses=n_poses, pool_choice=pool)

    print("\n" + "=" * 60)
    print("✅ Stage 6 complete.")
    print(f"   Outputs saved to : {config.STAGE6_DIR}")
    print(f"   Summary CSV      : {os.path.join(config.STAGE6_DIR, 'docking_summary.csv')}")
    print(f"   Ligands processed: {len(results)}")
    for lig_id, r in sorted(results.items()):
        print(f"   • {lig_id}  docked={r['n_docked']}  "
              f"failed={r['n_failed']}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
stage1_mask_calculation.py
==========================
Stage 1 of the pipeline:

    PDB file  +  PLIP XML  →  run_pipeline()  →  .meta.json files

Each .meta.json contains:
    • smiles                 – canonical SMILES of the ligand
    • selfies                – unmasked SELFIES encoding
    • masked                 – masked SELFIES string (BPE-adapted when enabled)
    • masked_smiles          – masked SMILES string (BPE-adapted, ChemBERTa-ready)
    • masked_atom_indices    – 0-based atom indices selected for masking (PLIP)
    • bpe_adapted_atom_indices / bpe_mask_count — SMILES BPE adapter metadata
    • masked_atoms_detail    – per-atom interaction metadata

These JSON files are the handshake with Stage 2.

Run standalone:
    python stage1_mask_calculation.py
"""

import json, os, glob, re, tempfile, subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, Set, Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, GetSymmSSSR
from rdkit.Chem.Draw import rdMolDraw2D
import selfies as sf

import config


# ════════════════════════════════════════════════════════════════════════════
#  PDB UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def extract_ligand_pdb(pdb_in: str, pdb_out: str,
                       resname: str, chain: Optional[str] = None,
                       resseq: Optional[int] = None):
    """Extract a single ligand from a full PDB file into a ligand-only PDB."""
    serials, detailed, ligand_lines = [], [], []

    with open(pdb_in) as fin:
        for line in fin:
            if not (line.startswith('HETATM') or line.startswith('ATOM  ')):
                continue
            if line[17:20].strip() != resname:
                continue
            if chain and line[21].strip() != chain:
                continue
            try:
                lseq = int(line[22:26])
            except Exception:
                lseq = None
            if resseq is not None and lseq != resseq:
                continue
            ligand_lines.append(line)
            try:
                serial = int(line[6:11])
            except Exception:
                continue
            atom_name = line[12:16].strip()
            element   = line[76:78].strip() or atom_name[0]
            serials.append(serial)
            detailed.append((serial, atom_name, element))

    if not serials:
        raise ValueError("No atoms found for ligand selection.")

    serial_set   = set(serials)
    conect_lines = []
    with open(pdb_in) as fin:
        for line in fin:
            if not line.startswith('CONECT'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                main_serial = int(parts[1])
            except ValueError:
                continue
            if main_serial not in serial_set:
                continue
            connected = []
            for p in parts[2:]:
                try:
                    connected.append(int(p))
                except ValueError:
                    pass
            if connected and all(s in serial_set for s in connected):
                conect_lines.append(line)

    with open(pdb_out, 'w') as fout:
        for line in ligand_lines:
            fout.write(line)
        for line in conect_lines:
            fout.write(line)
        fout.write("END\n")
    return serials, detailed


def pdb_to_sdf_with_obabel(pdb_path: str, sdf_out: str) -> None:
    res = subprocess.run(
        ["obabel", "-ipdb", pdb_path, "-osdf", "-O", sdf_out, "-h"],
        capture_output=True, text=True
    )
    if res.returncode != 0 or not os.path.exists(sdf_out) or os.path.getsize(sdf_out) == 0:
        raise RuntimeError(f"OpenBabel pdb→sdf failed: {res.stderr.strip()}")


def sdf_to_smiles_with_obabel(sdf_path: str) -> str:
    res = subprocess.run(
        ["obabel", sdf_path, "-O", "-", "-osmi"],
        capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"OpenBabel sdf→smi failed: {res.stderr.strip()}")
    return (res.stdout or "").strip()


def smiles_from_pdb_with_rdkit(pdb_path: str) -> Optional[str]:
    """RDKit fallback when OpenBabel is unavailable or returns empty."""
    try:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def robust_smiles_from_ligand_pdb(ligand_pdb: str, tmp_sdf: str) -> str:
    """Try OpenBabel first, fall back to RDKit, then a hard-coded placeholder."""
    try:
        pdb_to_sdf_with_obabel(ligand_pdb, tmp_sdf)
        smi = sdf_to_smiles_with_obabel(tmp_sdf)
        if smi:
            return smi
    except Exception as e:
        print("[DEBUG] OpenBabel route failed:", e)
    print("Warning: OpenBabel failed. Trying RDKit fallback.")
    smi_rd = smiles_from_pdb_with_rdkit(ligand_pdb)
    if smi_rd:
        return smi_rd
    print("Warning: RDKit also failed. Using placeholder SMILES.")
    return "OCCOCCOCCOCCOCCO"


def build_serial_to_rdkit_index(ligand_pdb: str) -> Dict[int, int]:
    mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=False, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError("RDKit failed to load ligand PDB.")
    conf = mol.GetConformer()
    serial_to_coord: Dict[int, np.ndarray] = {}
    with open(ligand_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                s = int(line[6:11])
                c = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                serial_to_coord[s] = c
            except Exception:
                continue
    serial_to_idx: Dict[int, int] = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = np.array(conf.GetAtomPosition(idx))
        for serial, coord in serial_to_coord.items():
            if np.linalg.norm(pos - coord) < 0.15:
                serial_to_idx[serial] = idx
                break
    return serial_to_idx


# ════════════════════════════════════════════════════════════════════════════
#  PLIP XML PARSING
# ════════════════════════════════════════════════════════════════════════════

def parse_plip_xml_v2_select(xml_path: str, resname: str,
                              chain: Optional[str], resseq: Optional[int],
                              include: Optional[Set[str]] = None):
    """
    Parse a PLIP XML file and return:
        (serial_to_types, smiles_to_pdb, lig_smiles)
    for the single binding site matching resname/chain/resseq.
    """
    include = set(include) if include else {
        "hydrophobic", "hbond", "waterBridge", "saltBridge",
        "piStacking", "piCation", "halogen", "metal",
    }
    tree = ET.parse(xml_path)
    root = tree.getroot()

    target_bs = None
    for bs in root.findall(".//bindingsite"):
        ids = bs.find("./identifiers")
        if ids is None:
            continue
        hetid    = (ids.findtext("hetid")    or "").strip()
        ch       = (ids.findtext("chain")    or "").strip()
        pos_text = (ids.findtext("position") or "").strip()
        try:
            pos = int(pos_text) if pos_text else None
        except Exception:
            pos = None
        if (hetid == resname
                and (chain is None or ch == chain)
                and (resseq is None or pos == resseq)):
            target_bs = bs
            break

    if target_bs is None:
        raise ValueError(
            f"No PLIP bindingsite matches {resname}:{chain}:{resseq} in {xml_path}"
        )

    lig_smiles = (target_bs.findtext("./identifiers/smiles") or "").strip() or None

    wanted_blocks = {
        "hydrophobic_interactions": "hydrophobic",
        "hydrogen_bonds":           "hbond",
        "water_bridges":            "waterBridge",
        "salt_bridges":             "saltBridge",
        "pi_stacks":                "piStacking",
        "pi_cation_interactions":   "piCation",
        "halogen_bonds":            "halogen",
        "metal_complexes":          "metal",
    }
    wanted_blocks = {k: v for k, v in wanted_blocks.items() if v in include}

    interactions: Dict[int, Set[str]] = defaultdict(set)
    inter = target_bs.find("./interactions")
    if inter is not None:
        if "hydrogen_bonds" in wanted_blocks:
            for hb in inter.findall("./hydrogen_bonds/hydrogen_bond"):
                protisdon = (hb.findtext("protisdon") or "").strip().lower() in ("true", "1", "yes")
                donor    = hb.findtext("donoridx")    or hb.findtext("donor_idx")
                acceptor = hb.findtext("acceptoridx") or hb.findtext("acceptor_idx")
                lig_serial = acceptor if protisdon else donor
                if lig_serial:
                    try:
                        interactions[int(lig_serial)].add(
                            ("hbond", "acceptor" if protisdon else "donor")
                        )
                    except Exception:
                        pass

        if "hydrophobic_interactions" in wanted_blocks:
            for hy in inter.findall("./hydrophobic_interactions/hydrophobic_interaction"):
                ligc = hy.findtext("ligcarbonidx")
                if ligc:
                    try:
                        interactions[int(ligc)].add("hydrophobic")
                    except Exception:
                        pass

        if "water_bridges" in wanted_blocks:
            for wb in inter.findall("./water_bridges/water_bridge"):
                protisdon = (wb.findtext("protisdon") or "").strip().lower() in ("true", "1", "yes")
                donor    = wb.findtext("donor_idx")    or wb.findtext("donoridx")
                acceptor = wb.findtext("acceptor_idx") or wb.findtext("acceptoridx")
                lig_serial = acceptor if protisdon else donor
                if lig_serial:
                    try:
                        interactions[int(lig_serial)].add("waterBridge")
                    except Exception:
                        pass

        if "salt_bridges" in wanted_blocks:
            for sb in inter.findall("./salt_bridges/salt_bridge"):
                fid = f"{sb.findtext('resnr')}{sb.findtext('reschain')}"
                tag = f"saltBridge:{fid}"
                for idx in sb.findall("./lig_idx_list/idx"):
                    try:
                        interactions[int(idx.text.strip())].add(tag)
                    except Exception:
                        pass
                lig_idx = sb.findtext("lig_idx") or sb.findtext("ligand_idx")
                if lig_idx:
                    try:
                        interactions[int(lig_idx)].add(tag)
                    except Exception:
                        pass

        if "pi_stacks" in wanted_blocks:
            for ps in inter.findall("./pi_stacks/pi_stack"):
                fid = f"{ps.findtext('resnr')}{ps.findtext('reschain')}"
                tag = f"piStacking:{fid}"
                for idx in ps.findall("./lig_idx_list/idx"):
                    try:
                        interactions[int(idx.text.strip())].add(tag)
                    except Exception:
                        pass

        if "pi_cation_interactions" in wanted_blocks:
            for pc in inter.findall("./pi_cation_interactions/pi_cation_interaction"):
                fid = f"{pc.findtext('resnr')}{pc.findtext('reschain')}"
                tag = f"piCation:{fid}"
                for idx in pc.findall("./lig_idx_list/idx"):
                    try:
                        interactions[int(idx.text.strip())].add(tag)
                    except Exception:
                        pass

        if "halogen_bonds" in wanted_blocks:
            for hx in inter.findall("./halogen_bonds/halogen_bond"):
                donor    = hx.findtext("donoridx")    or hx.findtext("donor_idx")
                acceptor = hx.findtext("acceptoridx") or hx.findtext("acceptor_idx")
                lig_serial = donor or acceptor
                if lig_serial:
                    try:
                        interactions[int(lig_serial)].add("halogen")
                    except Exception:
                        pass

        if "metal_complexes" in wanted_blocks:
            for mc in inter.findall("./metal_complexes/metal_complex"):
                for idx in mc.findall("./lig_idx_list/idx"):
                    try:
                        interactions[int(idx.text.strip())].add("metal")
                    except Exception:
                        pass
                lig_idx = mc.findtext("lig_idx")
                if lig_idx:
                    try:
                        interactions[int(lig_idx)].add("metal")
                    except Exception:
                        pass

    smiles_to_pdb: Dict[int, int] = {}
    mp = target_bs.find(".//mappings/smiles_to_pdb")
    if mp is not None and mp.text:
        for pair in mp.text.strip().split(","):
            if ":" in pair:
                a, b = pair.split(":")
                try:
                    smiles_to_pdb[int(a.strip())] = int(b.strip())
                except Exception:
                    pass

    return dict(interactions), smiles_to_pdb, lig_smiles


# ════════════════════════════════════════════════════════════════════════════
#  SELFIES MASKING
# ════════════════════════════════════════════════════════════════════════════

def is_atom_selfies_token(token: str) -> bool:
    """Return True only for tokens that represent a physical atom."""
    structural = {'Branch', 'Ring', 'Expl', 'Bond', '/', '\\', '<mask>'}
    if not (token.startswith('[') and token.endswith(']')):
        return False
    return not any(ind in token for ind in structural)


def mask_atoms_in_selfies(smiles: str,
                          atom_indices_to_mask: Iterable[int],
                          mask_token: str = '<mask>',
                          use_bpe_adapter: Optional[bool] = None,
                          ) -> Tuple[str, str, str, List[int]]:
    """
    Convert SMILES → SELFIES, replace selected atom tokens with mask_token.

    When BPE adapter is enabled (default), expands atom masks to full BPE token
    spans using config.CHEMBERTA_SELFIES_MODEL (A5b).

    Returns (smiles, selfies, selfies_masked, sorted_original_atom_indices).
    """
    original_sorted = sorted(set(int(i) for i in atom_indices_to_mask))

    enabled = (_adapter_enabled()
               if use_bpe_adapter is None else use_bpe_adapter)
    if enabled:
        from bpe_mask_adapter import adapt_selfies_mask
        result = adapt_selfies_mask(
            smiles, original_sorted, mask_token=mask_token,
        )
        selfies_str = sf.encoder(smiles)
        return smiles, selfies_str, result.masked_string, original_sorted

    selfies_str = sf.encoder(smiles)
    tokens      = list(sf.split_selfies(selfies_str))
    mask_set    = set(original_sorted)

    atom_token_positions: Dict[int, int] = {}
    atom_counter = 0
    for i, tok in enumerate(tokens):
        if is_atom_selfies_token(tok):
            atom_token_positions[atom_counter] = i
            atom_counter += 1

    for idx in original_sorted:
        if idx not in atom_token_positions:
            raise IndexError(
                f"Mask index {idx} out of range "
                f"(max {max(atom_token_positions.keys()) if atom_token_positions else -1})."
            )

    tokens_masked = tokens[:]
    for idx in original_sorted:
        tokens_masked[atom_token_positions[idx]] = mask_token

    return smiles, selfies_str, ''.join(tokens_masked), original_sorted


def _adapter_enabled() -> bool:
    return bool(getattr(config, "BPE_MASK_ADAPTER_ENABLED", True))


# ════════════════════════════════════════════════════════════════════════════
#  SMILES-LEVEL MASKING  (also imported by Stage 2)
# ════════════════════════════════════════════════════════════════════════════

_BRACKET_ATOM_RE = re.compile(r"\[[^\]]+\]")
_MAPNUM_RE       = re.compile(r":(\d+)\]$")

# Organic-subset atoms that appear unbracketed in SMILES.
_ORGANIC_ONE_CHAR = frozenset("BCNOPSFIbcnops")


def _clean_masking_enabled() -> bool:
    return bool(getattr(config, "CLEAN_SMILES_MASKING", False))


def _smiles_output_atom_order(mol: Chem.Mol) -> List[int]:
    """RDKit atom indices in the order they appear in MolToSmiles output."""
    prop = "_smilesAtomOutputOrder"
    if mol.HasProp(prop):
        raw = mol.GetProp(prop).strip()
        if raw.startswith("["):
            raw = raw[1: raw.rfind("]")]
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return list(range(mol.GetNumAtoms()))


def _clean_smiles_atom_spans(smiles: str) -> List[Tuple[int, int]]:
    """
    Return (start, end) character spans for each atom token in a *clean*
    (unmapped) SMILES string, in left-to-right output order.

    Handles bracket atoms ([...]), two-char organics (Cl, Br) and the
    unbracketed organic subset. Ring digits, bonds, branches, '%', '.', etc.
    are skipped (they are not atoms).
    """
    spans: List[Tuple[int, int]] = []
    i, n = 0, len(smiles)
    while i < n:
        ch = smiles[i]
        if ch == "[":
            end = smiles.index("]", i)
            spans.append((i, end + 1))
            i = end + 1
        elif ch == "C" and i + 1 < n and smiles[i + 1] == "l":
            spans.append((i, i + 2)); i += 2
        elif ch == "B" and i + 1 < n and smiles[i + 1] == "r":
            spans.append((i, i + 2)); i += 2
        elif ch in _ORGANIC_ONE_CHAR:
            spans.append((i, i + 1)); i += 1
        else:
            i += 1
    return spans


def mask_atoms_in_smiles_clean(smiles: str,
                               atom_indices_to_mask: Iterable[int],
                               tokenizer,
                               ) -> str:
    """
    Mask the requested 0-based atom indices, producing a CLEAN SMILES string
    where every unmasked atom keeps ChemBERTa's native form (bare 'C', 'c',
    'O', ...) instead of the bracketed '[CH3]', '[cH]' artifacts created by the
    atom-map round-trip.

    Each masked atom becomes exactly one tokenizer.mask_token (1 atom -> 1
    <mask>), matching the assumption of generate_smiles_sequential.

    Raises ValueError if the SMILES is invalid or atom/token counts disagree;
    callers fall back to the legacy path in that case.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mask_token = tokenizer.mask_token
    mask_set   = {int(i) for i in atom_indices_to_mask}

    clean = Chem.MolToSmiles(mol, canonical=True)
    order = _smiles_output_atom_order(mol)
    spans = _clean_smiles_atom_spans(clean)

    if len(spans) != len(order):
        raise ValueError(
            f"Atom-span count ({len(spans)}) != atom count ({len(order)}) "
            f"for SMILES {clean!r}"
        )

    # Map original atom index -> char span in the clean output string.
    idx_to_span = {order[k]: spans[k] for k in range(len(order))}

    mask_spans = sorted(idx_to_span[i] for i in mask_set if i in idx_to_span)
    if not mask_spans:
        return clean

    parts: List[str] = []
    pos = 0
    for start, end in mask_spans:
        parts.append(clean[pos:start])
        parts.append(mask_token)
        pos = end
    parts.append(clean[pos:])
    return "".join(parts)


def mask_atoms_in_smiles(smiles: str,
                         atom_indices_to_mask: Iterable[int],
                         tokenizer,
                         use_bpe_adapter: Optional[bool] = None,
                         ) -> str:
    """
    Replace the specified 0-based atom indices in *smiles* with
    tokenizer.mask_token (e.g. '<mask>').

    When BPE adapter is enabled (default), atom indices are expanded to full
    ChemBERTa BPE token spans (one <mask> per BPE token) before encoding.

    This function is imported by stage2_molecule_generation.py.
    """
    enabled = (_adapter_enabled()
               if use_bpe_adapter is None else use_bpe_adapter)
    if enabled:
        from bpe_mask_adapter import adapt_smiles_mask
        result = adapt_smiles_mask(
            smiles, atom_indices_to_mask, tokenizer=tokenizer,
        )
        return result.masked_string

    # Clean masking: bare atoms everywhere except the masked positions.
    if _clean_masking_enabled():
        try:
            return mask_atoms_in_smiles_clean(
                smiles, atom_indices_to_mask, tokenizer,
            )
        except Exception:
            pass   # fall through to the legacy bracketed path on any failure

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mask_set   = set(atom_indices_to_mask)
    mask_token = tokenizer.mask_token

    # Assign 1-based atom-map numbers so atom 0 can be identified in the string
    mol_mapped = Chem.Mol(mol)
    for a in mol_mapped.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)

    mapped = Chem.MolToSmiles(mol_mapped, canonical=False)

    out, last = [], 0
    for m in _BRACKET_ATOM_RE.finditer(mapped):
        out.append(mapped[last:m.start()])
        tok = m.group(0)
        mm  = _MAPNUM_RE.search(tok)
        if mm:
            atom_idx = int(mm.group(1)) - 1   # back to 0-based
            out.append(mask_token if atom_idx in mask_set else tok)
        else:
            out.append(tok)
        last = m.end()
    out.append(mapped[last:])

    masked = "".join(out)
    masked = re.sub(r":\d+(?=\])", "", masked)   # strip remaining map numbers
    return masked


# ════════════════════════════════════════════════════════════════════════════
#  2D INTERACTION VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def generate_2d_interaction_plot(results: Dict[str, Any], output_path: str) -> None:
    smiles              = results['smiles']
    masked_indices      = set(results['masked_atom_indices'])
    interactions_detail = results['masked_atoms_detail']
    mode                = results['masking_mode']

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError("RDKit failed to parse SMILES for visualization.")
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)

    canvas_size = 800
    drawer = rdMolDraw2D.MolDraw2DCairo(canvas_size, canvas_size)
    opts   = drawer.drawOptions()
    opts.padding       = 0.2
    opts.bondLineWidth = 3.0

    highlight_atoms, highlight_colors = [], {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in masked_indices:
            highlight_atoms.append(idx)
            highlight_colors[idx] = (1.0, 0.7, 0.7, 0.6)
            opts.atomLabels[idx]  = f"{atom.GetSymbol()}*"

    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                        highlightAtomColors=highlight_colors)
    drawer.FinishDrawing()

    coord_map: Dict[int, np.ndarray] = {}
    for i in range(mol.GetNumAtoms()):
        pt = drawer.GetDrawCoords(i)
        coord_map[i] = np.array([pt.x, canvas_size - pt.y])

    ring_centroids = {}
    for ring in GetSymmSSSR(mol):
        atoms = list(ring)
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            ring_centroids[frozenset(atoms)] = np.mean(
                [coord_map[i] for i in atoms], axis=0
            )

    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    ax.imshow(Image.open(io.BytesIO(drawer.GetDrawingText())),
              extent=[0, canvas_size, 0, canvas_size], zorder=0)

    color_map = {
        'hydrophobic': 'grey',   'hbond': 'blue',      'waterBridge': 'cyan',
        'saltBridge': 'yellow',  'piStacking': 'green', 'piCation': 'orange',
        'halogen': 'teal',       'metal': 'purple',
    }
    style_map = {
        'hydrophobic': 'dashed', 'hbond': 'solid',     'piStacking': 'dashdot',
        'piCation': 'dashdot',   'halogen': 'dotted',   'saltBridge': (0, (5, 5)),
        'waterBridge': 'solid',  'metal': 'solid',
    }

    def _norm(t):
        return t[0] if isinstance(t, tuple) else t

    center = np.array([canvas_size / 2, canvas_size / 2])
    interaction_groups: Dict[str, List[np.ndarray]] = {}

    for detail in interactions_detail:
        atom_idx = detail["masked_smiles_index_0b"]
        if atom_idx not in coord_map:
            continue
        for itype in detail["interaction_types"]:
            itype_name = _norm(itype)
            itype_role = itype[1] if isinstance(itype, tuple) and len(itype) > 1 else None
            anchor_idx = atom_idx
            atom       = mol.GetAtomWithIdx(anchor_idx)

            # Correct H-bond anchor away from halogens/carbons
            if itype_name == "hbond" and atom.GetSymbol() in ("F", "Cl", "Br", "I", "C"):
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() in ("N", "O", "S"):
                        if itype_role == "donor" and nbr.GetTotalNumHs() > 0:
                            anchor_idx = nbr.GetIdx(); break
                        elif itype_role in ("acceptor", None):
                            anchor_idx = nbr.GetIdx(); break

            anchor_point = coord_map[anchor_idx]

            # Pi interactions anchor on ring centroid
            if itype_name.lower() in ("pistacking", "pication"):
                for ring_atoms, centroid in ring_centroids.items():
                    if anchor_idx in ring_atoms:
                        anchor_point = centroid; break

            key = itype_name if ":" in itype_name else f"{itype_name}#{atom_idx}"
            interaction_groups.setdefault(key, []).append(anchor_point)

    bubbles = []
    for key, points in interaction_groups.items():
        anchor_point = np.array(points).mean(axis=0)
        if ":" in key:
            base_type  = key.split(":")[0]
            label_text = f"{base_type}\n{key.split(':')[1]}"
        else:
            base_type  = key.split("#")[0]
            label_text = base_type
        direction = anchor_point - center
        norm = np.linalg.norm(direction)
        direction /= (norm + 1e-8)
        dist       = np.clip(150 + (300 - norm) * 0.5, 100, 200)
        bubble_xy  = np.clip(anchor_point + direction * dist, 50, canvas_size - 50)
        bubbles.append((bubble_xy, key, base_type, label_text, anchor_point))

    # Collision avoidance
    for _ in range(20):
        for i in range(len(bubbles)):
            for j in range(i + 1, len(bubbles)):
                d = np.linalg.norm(bubbles[i][0] - bubbles[j][0])
                if d < 50:
                    delta = (bubbles[i][0] - bubbles[j][0]) / (d + 1e-8)
                    p1, t1, b1, l1, a1 = bubbles[i]
                    p2, t2, b2, l2, a2 = bubbles[j]
                    bubbles[i] = (p1 + delta * 15, t1, b1, l1, a1)
                    bubbles[j] = (p2 - delta * 15, t2, b2, l2, a2)

    legend_handles: Dict[str, Any] = {}
    for pos, _, base_type, label_text, anchor in bubbles:
        color = color_map.get(base_type, "black")
        style = style_map.get(base_type, "solid")
        ax.plot([anchor[0], pos[0]], [anchor[1], pos[1]],
                color=color, linestyle=style, linewidth=2, alpha=0.85)
        ax.add_patch(plt.Circle(pos, 35, facecolor="#CCFFCC", edgecolor=color, lw=2))
        ax.text(pos[0], pos[1], label_text,
                ha="center", va="center", fontsize=8, fontweight="bold")
        if base_type not in legend_handles:
            legend_handles[base_type] = mpatches.Patch(color=color, label=base_type)

    masked_patch = mpatches.Patch(facecolor=(1.0, 0.7, 0.7, 0.6),
                                  edgecolor="red", label="Masked Atom (*)")
    ax.legend(handles=[masked_patch] + list(legend_handles.values()),
              loc="upper right", title=f"Mode: {mode.upper()}",
              framealpha=0.9, fancybox=True, shadow=True)
    ax.set_xlim(0, canvas_size); ax.set_ylim(0, canvas_size); ax.axis("off")
    plt.title(f"2D Interaction Audit: {results['ligand']['resname']}",
              fontsize=16, fontweight="bold")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(pdb_path: str,
                 plip_xml_path: str,
                 resname: str,
                 chain: Optional[str],
                 resseq: Optional[int],
                 include_types: Iterable[str],
                 representation: str = 'selfies',
                 mask_token: str = '<mask>',
                 out_prefix: Optional[str] = None,
                 serial_map_json: Optional[str] = None,
                 mask_non_attractive: bool = False) -> Dict[str, Any]:
    """
    Full Stage-1 pipeline:
        PDB + PLIP XML  →  meta dict  (+ optional .meta.json + .png)

    The returned dict is the 'meta' object consumed by Stage 2.
    """
    # 1) Parse PLIP interactions
    serial_to_types, smiles2pdb, plip_smiles = parse_plip_xml_v2_select(
        plip_xml_path, resname=resname, chain=chain, resseq=resseq,
        include=set(include_types)
    )
    print("SERIAL TO TYPES:", serial_to_types)
    if not serial_to_types and not mask_non_attractive:
        raise ValueError("No interactions found in PLIP XML for the requested ligand/types.")

    with tempfile.TemporaryDirectory() as td:
        lig_pdb = os.path.join(td, "ligand.pdb")
        serials_in_order, _ = extract_ligand_pdb(
            pdb_path, lig_pdb, resname=resname, chain=chain, resseq=resseq
        )
        lig_sdf = os.path.join(td, "ligand.sdf")
        smiles  = robust_smiles_from_ligand_pdb(lig_pdb, lig_sdf)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError("RDKit failed to parse SMILES.")

        serial_to_idx = (
            {int(k): int(v) for k, v in json.load(open(serial_map_json)).items()}
            if serial_map_json
            else build_serial_to_rdkit_index(lig_pdb)
        )

        # Robust serial → SMILES-index mapping via atom-map numbers
        ref_mol_pdb = Chem.MolFromPDBFile(lig_pdb, removeHs=True, sanitize=False)
        serial_to_ref_idx: Dict[int, int] = {}
        if ref_mol_pdb:
            ref_conf = ref_mol_pdb.GetConformer()
            serial_to_coord_map: Dict[int, np.ndarray] = {}
            with open(lig_pdb) as f_pdb:
                for line in f_pdb:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue
                    try:
                        s_ = int(line[6:11])
                        c_ = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        serial_to_coord_map[s_] = c_
                    except Exception:
                        pass
            for atom in ref_mol_pdb.GetAtoms():
                idx_ = atom.GetIdx()
                pos  = np.array(ref_conf.GetAtomPosition(idx_))
                best_s, best_dist = None, 0.15
                for s_, c_ in serial_to_coord_map.items():
                    d_ = np.linalg.norm(pos - c_)
                    if d_ < best_dist:
                        best_dist = d_; best_s = s_
                if best_s is not None:
                    serial_to_ref_idx[best_s] = idx_

        ref_mol_fixed = ref_mol_pdb
        if plip_smiles and ref_mol_pdb:
            try:
                tmpl = Chem.MolFromSmiles(plip_smiles)
                if tmpl:
                    ref_mol_fixed = AllChem.AssignBondOrdersFromTemplate(tmpl, ref_mol_pdb)
                    Chem.SanitizeMol(ref_mol_fixed)
                    Chem.Kekulize(ref_mol_fixed, clearAromaticFlags=True)
            except Exception:
                ref_mol_fixed = ref_mol_pdb

        for atom in ref_mol_fixed.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        mapped_smiles = Chem.MolToSmiles(ref_mol_fixed, isomericSmiles=True)
        target_mol    = Chem.MolFromSmiles(mapped_smiles)

        ref_idx_to_target_idx: Dict[int, int] = {}
        if target_mol:
            for atom in target_mol.GetAtoms():
                mn = atom.GetAtomMapNum()
                if mn > 0:
                    ref_idx_to_target_idx[mn - 1] = atom.GetIdx()
            for atom in target_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            smiles     = Chem.MolToSmiles(target_mol, isomericSmiles=True, canonical=False)
            target_mol = Chem.MolFromSmiles(smiles)

        serial_to_smiles_idx = {
            s: ref_idx_to_target_idx[r_idx]
            for s, r_idx in serial_to_ref_idx.items()
            if r_idx in ref_idx_to_target_idx
        }

    # 2) Determine which atom indices to mask
    attractive_indices = sorted({
        serial_to_smiles_idx[s]
        for s in serial_to_types
        if s in serial_to_smiles_idx
    })

    if mask_non_attractive:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError("RDKit failed to parse SMILES for inversion.")
        masked_indices = sorted(set(range(mol.GetNumAtoms())) - set(attractive_indices))
    else:
        masked_indices = attractive_indices

    if not masked_indices:
        raise ValueError("Resolved 0 mask indices. Check XML or ligand selection.")

    # 3) SELFIES masking
    if representation.lower() != 'selfies':
        raise NotImplementedError("Only SELFIES masking is implemented in Stage 1.")

    smiles_in, selfies_in, selfies_masked, masked_sorted = mask_atoms_in_selfies(
        smiles, masked_indices, mask_token=mask_token
    )

    from bpe_mask_adapter import build_smiles_mask_json_fields
    smiles_mask_fields = build_smiles_mask_json_fields(smiles, masked_sorted)

    # 4) Build per-atom detail metadata
    tokens_in = list(sf.split_selfies(selfies_in))
    try:
        tokens_masked = list(sf.split_selfies(selfies_masked))
    except Exception:
        # BPE-adapted strings may contain literal <mask> spans (not valid SELFIES)
        tokens_masked = []
    atom_token_index_map: Dict[int, int] = {}
    atom_counter = 0
    for i, tok in enumerate(tokens_in):
        if is_atom_selfies_token(tok):
            atom_token_index_map[atom_counter] = i
            atom_counter += 1

    idx_to_serial = {v: k for k, v in serial_to_smiles_idx.items()}

    def _norm(t):
        return t[0] if isinstance(t, tuple) else t

    masked_atoms_detail = []
    for _idx0 in masked_sorted:
        _pdb_serial = idx_to_serial.get(_idx0)
        token_pos   = atom_token_index_map.get(_idx0)
        _types = sorted({
            t for t in (_norm(t) for t in serial_to_types.get(_pdb_serial, []))
            if t
        }) if _pdb_serial is not None else []
        masked_atoms_detail.append({
            "masked_smiles_index_0b": _idx0,
            "selfies_token_index":    token_pos,
            "original_token": tokens_in[token_pos]     if (token_pos is not None and token_pos < len(tokens_in))     else None,
            "masked_token":   tokens_masked[token_pos] if (token_pos is not None and token_pos < len(tokens_masked)) else None,
            "pdb_serial":        _pdb_serial,
            "interaction_types": _types,
        })

    meta = {
        "smiles":              smiles,
        "selfies":             selfies_in,
        "masked":              selfies_masked,
        "mask_token":          mask_token,
        "masked_atom_indices": masked_sorted,
        **smiles_mask_fields,
        "include_types":       list(include_types),
        "masking_mode":        "non-attractive" if mask_non_attractive else "attractive",
        "pdb":                 pdb_path,
        "plip_xml":            plip_xml_path,
        "ligand":              {"resname": resname, "chain": chain, "resseq": resseq},
        "masked_atoms_detail": masked_atoms_detail,
    }

    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        json_path = (os.path.splitext(out_prefix)[0]
                     + f"_{resname}_{chain}_{resseq}.meta.json")
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  💾 JSON saved: {json_path}")

        viz_path = os.path.splitext(out_prefix)[0] + ".2d_interactions.png"
        generate_2d_interaction_plot(meta, viz_path)
        print(f"  🖼  Plot saved: {viz_path}")

    return meta


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def _select_pipeline_inputs() -> List[dict]:
    """
    Prompt the user to choose which ligand(s) from config.PIPELINE_INPUTS to
    process: all of them, or a comma-separated subset selected by number.

    Re-prompts until a valid choice is entered. Returns the selected input
    dicts in menu order (duplicates removed).
    """
    inputs = list(config.PIPELINE_INPUTS)
    if not inputs:
        return []

    print("\n  Available ligands:")
    for i, inp in enumerate(inputs, start=1):
        tag = f"{inp['resname']}_{inp['chain']}_{inp['resseq']}"
        print(f"    {i:>2d} : {tag:<16s}  (pdb: {inp['pdb_path']})")
    print("    all : run every ligand listed above  [default]")

    while True:
        raw = input(
            "\n  Select ligand(s) — comma-separated numbers (e.g. 1,3) "
            "or 'all' [all]: "
        ).strip().lower()

        if raw in ("", "all"):
            print(f"\n  ✅ Selected: ALL {len(inputs)} ligand(s).")
            return inputs

        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        try:
            picks = [int(t) for t in tokens]
        except ValueError:
            print("    Please enter numbers separated by commas, or 'all'.")
            continue

        if not picks:
            print("    Please enter at least one number, or 'all'.")
            continue

        if any(p < 1 or p > len(inputs) for p in picks):
            print(f"    Numbers must be between 1 and {len(inputs)}.")
            continue

        seen: Set[int] = set()
        selected: List[dict] = []
        for i, inp in enumerate(inputs, start=1):
            if i in picks and i not in seen:
                seen.add(i)
                selected.append(inp)

        tags = ", ".join(
            f"{inp['resname']}_{inp['chain']}_{inp['resseq']}" for inp in selected
        )
        print(f"\n  ✅ Selected {len(selected)} ligand(s): {tags}")
        return selected


def main():
    os.makedirs(config.MASK_CALC_OUTDIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("STAGE 1: MASK CALCULATION")
    print("=" * 60)

    # ── Ask masking mode at runtime ───────────────────────────────────────────
    print("""
  Masking mode:
    1 : INTERACTION      (mask_non_attractive = False)  [default]
        Atoms that DO participate in protein-ligand interactions
        (hydrophobic, H-bond, pi-stacking, salt bridge, etc.) are masked.
        ChemBERTa reconstructs the interacting atoms — generates molecules
        with alternative binding groups, scaffold unchanged.

    2 : NON-INTERACTION  (mask_non_attractive = True)
        Atoms that do NOT participate in interactions are masked.
        ChemBERTa reconstructs the non-interacting scaffold — generates
        molecules with alternative scaffolds, binding pharmacophore unchanged.
""")
    while True:
        _mode_raw = input("  Select masking mode (1 / 2) [1]: ").strip()
        if _mode_raw in ("", "1"):
            _mask_non_attractive = False
            print("""
  ✅ Masking mode : INTERACTION  (mask_non_attractive = False)
  Interacting atoms will be masked.
""")
            break
        elif _mode_raw == "2":
            _mask_non_attractive = True
            print("""
  ✅ Masking mode : NON-INTERACTION  (mask_non_attractive = True)
  Non-interacting atoms will be masked.
""")
            break
        else:
            print("    Please type 1 or 2.")

    selected_inputs = _select_pipeline_inputs()
    if not selected_inputs:
        print("\n  ❌ No ligands selected / configured. Exiting.")
        return

    for inp in selected_inputs:
        tag = f"{inp['resname']}_{inp['chain']}_{inp['resseq']}"
        print(f"\n→ Processing {tag}")
        try:
            run_pipeline(
                pdb_path            = inp["pdb_path"] + ".pdb",
                plip_xml_path       = inp["plip_xml_path"] + ".xml",
                resname             = inp["resname"],
                chain               = inp["chain"],
                resseq              = inp["resseq"],
                include_types       = config.INCLUDE_TYPES,
                representation      = 'selfies',
                mask_token          = '<mask>',
                out_prefix          = os.path.join(
                    config.MASK_CALC_OUTDIR, tag + "_masked.selfies"
                ),
                serial_map_json     = None,
                mask_non_attractive = _mask_non_attractive,
            )
        except Exception as e:
            print(f"  ⚠️  Skipped {tag}: {e}")

    print("\n✅ Stage 1 complete.")
    print(f"   JSON files saved to: {config.MASK_CALC_OUTDIR}")
    print("   Run stage2_molecule_generation.py next.")


if __name__ == "__main__":
    main()

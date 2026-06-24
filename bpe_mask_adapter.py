# -*- coding: utf-8 -*-
"""
bpe_mask_adapter.py
===================
BPE-aligned masking for ChemBERTa (SMILES and SELFIES).

Expands RDKit atom-index masks so each replacement is exactly one ChemBERTa
BPE token (A1a token-cover, A2a full token span, A3a one <mask> per BPE token).

Used centrally by mask_atoms_in_smiles() and mask_atoms_in_selfies() in
stage1_mask_calculation.py (A4a).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Any

from rdkit import Chem

import config

try:
    import selfies as sf
except ImportError:
    sf = None  # type: ignore

try:
    from transformers import PreTrainedTokenizerBase
except ImportError:
    PreTrainedTokenizerBase = object  # type: ignore

_ORGANIC_ATOM_CHARS = frozenset("BCNOFPSbcnops")
_BRACKET_ATOM_RE = re.compile(r"\[[^\]]+\]")
_SELFIES_STRUCTURAL = frozenset({"Branch", "Ring", "Expl", "Bond", "/", "\\", "<mask>"})


def _is_atom_selfies_token(token: str) -> bool:
    if not (token.startswith("[") and token.endswith("]")):
        return False
    return not any(ind in token for ind in _SELFIES_STRUCTURAL)


@dataclass
class BPEAdaptedMask:
    """Result of BPE mask adaptation."""

    original_atom_indices: List[int]
    adapted_atom_indices:  List[int]
    masked_string:         str
    representation:        str          # "smiles" | "selfies"
    bpe_mask_count:        int = 0
    bpe_token_spans:       List[Tuple[int, int]] = field(default_factory=list)


def _adapter_enabled() -> bool:
    return bool(getattr(config, "BPE_MASK_ADAPTER_ENABLED", True))


@lru_cache(maxsize=2)
def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def get_smiles_bpe_tokenizer():
    """ChemBERTa tokenizer for SMILES (zinc-base-v1)."""
    return _load_tokenizer(config.CHEMBERTA_MODEL)


def get_selfies_bpe_tokenizer():
    """ChemBERTa tokenizer for SELFIES (must be trained on SELFIES)."""
    model = getattr(config, "CHEMBERTA_SELFIES_MODEL", None)
    if not model:
        raise ValueError(
            "config.CHEMBERTA_SELFIES_MODEL is not set. "
            "Required for SELFIES BPE adaptation (A5b)."
        )
    return _load_tokenizer(model)


def _smiles_atom_output_order(mol: Chem.Mol) -> List[int]:
    """RDKit atom order as they appear in MolToSmiles output."""
    m = Chem.Mol(mol)
    Chem.MolToSmiles(m, canonical=False)
    prop = "_smilesAtomOutputOrder"
    if m.HasProp(prop):
        raw = m.GetProp(prop).strip()
        # RDKit stores e.g. "[0,1,2,...]" — not bare comma-separated ints
        if raw.startswith("["):
            raw = raw[1 : raw.rfind("]")]
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return list(range(m.GetNumAtoms()))


def build_smiles_atom_char_spans(mol: Chem.Mol) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    """
    Map each RDKit atom index to a character span in an atom-mapped SMILES string.
    """
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    mapped = Chem.MolToSmiles(m, canonical=False)
    order  = _smiles_atom_output_order(m)
    spans: Dict[int, Tuple[int, int]] = {}

    atom_out_i = 0
    i = 0
    while i < len(mapped) and atom_out_i < len(order):
        if mapped[i] == "[":
            end = mapped.index("]", i)
            atom_idx = order[atom_out_i]
            spans[atom_idx] = (i, end + 1)
            i = end + 1
            atom_out_i += 1
        elif mapped[i] in _ORGANIC_ATOM_CHARS:
            atom_idx = order[atom_out_i]
            spans[atom_idx] = (i, i + 1)
            i += 1
            atom_out_i += 1
        else:
            i += 1

    return mapped, spans


def build_selfies_atom_char_spans(smiles: str) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    """
    Map each RDKit atom index to a character span in the SELFIES string.
    """
    if sf is None:
        raise ImportError("selfies is required for SELFIES BPE adaptation")

    selfies_str = sf.encoder(smiles)
    tokens      = list(sf.split_selfies(selfies_str))
    spans: Dict[int, Tuple[int, int]] = {}

    pos = 0
    atom_counter = 0
    for tok in tokens:
        start, end = pos, pos + len(tok)
        if _is_atom_selfies_token(tok):
            spans[atom_counter] = (start, end)
            atom_counter += 1
        pos = end

    return selfies_str, spans


def _merge_char_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    sorted_spans = sorted(spans)
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _char_to_atom_map(
    spans: Dict[int, Tuple[int, int]],
    string_len: int,
) -> Dict[int, int]:
    char_map: Dict[int, int] = {}
    for atom_idx, (start, end) in spans.items():
        for c in range(start, min(end, string_len)):
            char_map[c] = atom_idx
    return char_map


def _tokenize_with_offsets(tokenizer, text: str) -> List[Tuple[int, int]]:
    """Return list of (start, end) char spans per BPE token (no special tokens)."""
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"]
    return [(int(s), int(e)) for s, e in offsets if int(s) < int(e)]


def expand_atoms_for_bpe_tokens(
    string: str,
    atom_char_spans: Dict[int, Tuple[int, int]],
    seed_atom_indices: Iterable[int],
    tokenizer,
) -> Tuple[Set[int], List[Tuple[int, int]]]:
    """
    A1a + A2a: iteratively expand atom set until all touched BPE tokens are
    fully covered; return merged char spans (one <mask> per span, A3a).
    """
    seed: Set[int] = set(seed_atom_indices)
    if not seed:
        return set(), []

    char_to_atom = _char_to_atom_map(atom_char_spans, len(string))
    bpe_offsets  = _tokenize_with_offsets(tokenizer, string)

    adapted_atoms: Set[int] = set(seed)
    mask_spans: List[Tuple[int, int]] = []

    changed = True
    while changed:
        changed = False
        for ts, te in bpe_offsets:
            touched_atoms = {
                char_to_atom[c]
                for c in range(ts, te)
                if c in char_to_atom and char_to_atom[c] in adapted_atoms
            }
            if not touched_atoms:
                continue

            full_atoms = {
                char_to_atom[c]
                for c in range(ts, te)
                if c in char_to_atom
            }
            if full_atoms - adapted_atoms:
                adapted_atoms |= full_atoms
                changed = True

            mask_spans.append((ts, te))

    return adapted_atoms, _merge_char_spans(mask_spans)


def apply_bpe_mask_spans(
    string: str,
    spans: Sequence[Tuple[int, int]],
    mask_token: str,
) -> str:
    """Replace each char span with a single mask_token (A3a)."""
    if not spans:
        return string

    parts: List[str] = []
    pos = 0
    for start, end in sorted(spans):
        parts.append(string[pos:start])
        parts.append(mask_token)
        pos = end
    parts.append(string[pos:])
    return "".join(parts)


def _verify_bpe_masks(tokenizer, masked: str, mask_token: str) -> int:
    """Count <mask> token IDs after tokenization (sanity check)."""
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        return masked.count(mask_token)
    enc = tokenizer(masked, add_special_tokens=False)
    return sum(1 for tid in enc["input_ids"] if tid == mask_id)


def adapt_smiles_mask(
    smiles: str,
    atom_indices: Iterable[int],
    tokenizer=None,
    mask_token: Optional[str] = None,
) -> BPEAdaptedMask:
    """BPE-adapt a SMILES atom mask for ChemBERTa-zinc-base-v1."""
    if tokenizer is None:
        tokenizer = get_smiles_bpe_tokenizer()
    if mask_token is None:
        mask_token = tokenizer.mask_token

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    original = sorted(set(int(i) for i in atom_indices))
    mapped, atom_spans = build_smiles_atom_char_spans(mol)

    adapted_atoms, bpe_spans = expand_atoms_for_bpe_tokens(
        mapped, atom_spans, original, tokenizer,
    )
    masked = apply_bpe_mask_spans(mapped, bpe_spans, mask_token)
    masked = re.sub(r":\d+(?=\])", "", masked)

    return BPEAdaptedMask(
        original_atom_indices = original,
        adapted_atom_indices  = sorted(adapted_atoms),
        masked_string         = masked,
        representation        = "smiles",
        bpe_mask_count        = _verify_bpe_masks(tokenizer, masked, mask_token),
        bpe_token_spans       = list(bpe_spans),
    )


def adapt_selfies_mask(
    smiles: str,
    atom_indices: Iterable[int],
    tokenizer=None,
    mask_token: Optional[str] = None,
) -> BPEAdaptedMask:
    """BPE-adapt a SELFIES atom mask for a SELFIES ChemBERTa tokenizer."""
    if tokenizer is None:
        tokenizer = get_selfies_bpe_tokenizer()
    if mask_token is None:
        mask_token = tokenizer.mask_token

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    original = sorted(set(int(i) for i in atom_indices))
    selfies_str, atom_spans = build_selfies_atom_char_spans(smiles)

    adapted_atoms, bpe_spans = expand_atoms_for_bpe_tokens(
        selfies_str, atom_spans, original, tokenizer,
    )
    masked = apply_bpe_mask_spans(selfies_str, bpe_spans, mask_token)

    return BPEAdaptedMask(
        original_atom_indices = original,
        adapted_atom_indices  = sorted(adapted_atoms),
        masked_string         = masked,
        representation        = "selfies",
        bpe_mask_count        = _verify_bpe_masks(tokenizer, masked, mask_token),
        bpe_token_spans       = list(bpe_spans),
    )


def adapt_atom_indices_for_bpe(
    smiles: str,
    atom_indices: Iterable[int],
    representation: str = "smiles",
    tokenizer=None,
) -> BPEAdaptedMask:
    """
    Public entry point for stages 1 / 1.5 / 1.7 to inspect adapted indices
    without building downstream strings.
    """
    rep = representation.lower()
    if rep == "smiles":
        return adapt_smiles_mask(smiles, atom_indices, tokenizer=tokenizer)
    if rep == "selfies":
        return adapt_selfies_mask(smiles, atom_indices, tokenizer=tokenizer)
    raise ValueError(f"representation must be 'smiles' or 'selfies', got {representation!r}")


def build_smiles_mask_json_fields(
    smiles: str,
    atom_indices: Iterable[int],
    tokenizer=None,
) -> Dict[str, Any]:
    """
    Central helper (S6c) for Stages 1, 1.5, and 1.7 JSON output.

    Builds BPE-adapted masked SMILES (S1a full mask, S2a, S3a JSON fields):
      • masked_smiles              — ChemBERTa-ready masked SMILES string
      • bpe_adapted_atom_indices   — atom indices after BPE token-cover expansion
      • bpe_mask_count             — number of <mask> tokens after BPE tokenization

    Loads config.CHEMBERTA_MODEL tokenizer on first call (S5a).
    """
    original = sorted(set(int(i) for i in atom_indices))

    if tokenizer is None:
        tokenizer = get_smiles_bpe_tokenizer()

    if not original:
        return {
            "masked_smiles":            smiles,
            "bpe_adapted_atom_indices":   [],
            "bpe_mask_count":             0,
        }

    if _adapter_enabled():
        result = adapt_smiles_mask(smiles, original, tokenizer=tokenizer)
        return {
            "masked_smiles":            result.masked_string,
            "bpe_adapted_atom_indices": result.adapted_atom_indices,
            "bpe_mask_count":           result.bpe_mask_count,
        }

    # Legacy per-atom masking (adapter disabled)
    from stage1_mask_calculation import mask_atoms_in_smiles
    masked = mask_atoms_in_smiles(
        smiles, original, tokenizer, use_bpe_adapter=False,
    )
    return {
        "masked_smiles":            masked,
        "bpe_adapted_atom_indices": original,
        "bpe_mask_count":           masked.count(tokenizer.mask_token),
    }

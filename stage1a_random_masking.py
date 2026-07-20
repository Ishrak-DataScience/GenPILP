# -*- coding: utf-8 -*-
"""
stage1a_random_masking.py
==========================
Stage 1a of the pipeline:

    Stage-0a verified ChEMBL SMILES  →  random TOKEN-level masking  →  ChemBERTa
    → exactly one generated molecule per (SMILES, mask_percent, temperature, seed)

Unlike Stage 1.5 / 1.7 (which sample a fraction of RDKit ATOMS and then expand
to covering BPE-token spans), Stage 1a masks a percentage of ChemBERTa BPE
TOKENS directly: the raw SMILES is tokenized, that percent of token positions
is chosen uniformly at random, and each chosen token span is replaced with
exactly one <mask>.

For every (SMILES, mask_percent, temperature, seed) combination, ChemBERTa
draws exactly ONE candidate molecule (num_samples=1) — not aggregated over
many draws like Stage 2 / 2.5 / 2.7.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN DECISIONS (resolved with the user; see conversation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
D1. Mask unit = % of BPE tokens (direct), not % of heavy atoms.
D2. config.STAGE1A_MASK_PERCENTS = [5, 10, 15, 20].
D3. config.STAGE1A_INPUT_LIMIT caps how many Stage-0a rows are processed —
    a uniform random sample of that size, drawn from the ENTIRE
    chembl_verified_smiles.csv via seeded reservoir sampling (representative
    of the full file, not just its first N rows). None = all rows, no
    sampling. See config.STAGE1A_INPUT_SAMPLE_SEED.
D4. Output is one aggregated CSV per (percent, temperature, seed) combo in
    config.STAGE1A_DIR, not one JSON file per molecule (avoids millions of
    tiny files at ChEMBL scale).
D5. Each seed deterministically controls BOTH which tokens get masked AND
    the ChemBERTa sampling draw (via torch.manual_seed), so re-running the
    same (molecule, percent, temperature, seed) always reproduces the same
    generated molecule.
D6. Summary plot's y-metric = (# distinct RDKit-CANONICAL valid generated
    SMILES) / (# molecules attempted), computed per (percent, temperature,
    seed) then averaged across seeds. One curve per temperature, colors
    assigned in a fixed order (never cycled/re-picked based on which
    temperatures are present).
D7. RDKit's native log chatter (parse errors/warnings) is captured per
    molecule and appended to <output_dir>/rdkit_errors.txt, tagged with the
    (chembl_id, percent, temperature, seed) that produced it, instead of
    printing over the tqdm progress bar (matches Stage 2.7's clean-console
    convention, but keeps the messages instead of discarding them).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETERMINISM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mask_seed       = sha256(seed:chembl_id:percent)        → which tokens masked
  generation_seed = sha256(seed:chembl_id:percent:temperature) → torch.manual_seed
  Both are independent of processing order, so a resumed run reproduces
  exactly the same output as a fresh full run.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESUMABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Each combo's CSV is read on start; chembl_ids already present are skipped
  and new rows are appended. Deleting a combo's CSV forces a full re-run of
  that combo only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage0a_chembl_2M_download.py   (once, to populate STAGE0A_DIR)
    python stage1a_random_masking.py

HOW TO TEST (quick smoke test, no full ChEMBL download needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage1a_random_masking.py --test
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from rdkit import rdBase as _rdBase

# Route RDKit's native C++ log messages (e.g. "SMILES Parse Error") through
# Python's sys.stderr instead of writing directly to the OS-level stderr fd,
# so contextlib.redirect_stderr can capture them per-molecule below (see
# run_stage1a) instead of them interleaving with the tqdm progress bar.
_rdBase.LogToPythonStderr()

import config
from bpe_mask_adapter import apply_bpe_mask_spans
from stage2_molecule_generation import generate_smiles, load_chemberta

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else range(0)


CSV_FIELDS = [
    "chembl_id", "smiles", "mask_percent", "temperature", "seed",
    "n_tokens", "n_masked", "masked_smiles", "generated_smiles", "valid",
]


# ════════════════════════════════════════════════════════════════════════════
#  INPUT LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_stage0a_smiles(
    csv_path: Optional[str] = None,
    limit: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Load (chembl_id, smiles) pairs from Stage 0a's verified SMILES CSV.

    When `limit` is set, a uniform random sample of that size is drawn from
    the ENTIRE file via reservoir sampling (Algorithm R) — a single pass,
    O(limit) memory, no need to know the row count up front — so the subset
    is representative of the full ChEMBL file rather than just its first N
    rows. The sample is seeded (sample_seed, else config.STAGE1A_INPUT_SAMPLE_SEED)
    so re-running with the same seed reproduces the same subset, then sorted
    by chembl_id for a stable, readable row order in the output CSVs.

    limit=None (or 0) returns every row in file order — no sampling.
    """
    csv_path = csv_path or os.path.join(config.STAGE0A_DIR, "chembl_verified_smiles.csv")

    if not limit:
        rows: List[Tuple[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["chembl_id"], row["smiles"]))
        return rows

    sample_seed = (
        sample_seed if sample_seed is not None
        else getattr(config, "STAGE1A_INPUT_SAMPLE_SEED", 42)
    )
    rng = random.Random(sample_seed)

    reservoir: List[Tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            item = (row["chembl_id"], row["smiles"])
            if i < limit:
                reservoir.append(item)
            else:
                j = rng.randint(0, i)
                if j < limit:
                    reservoir[j] = item

    reservoir.sort()
    return reservoir


# ════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC SEEDING (D5)
# ════════════════════════════════════════════════════════════════════════════

def mask_seed(base_seed: int, chembl_id: str, percent: float) -> int:
    """Per-(molecule, percent) seed for choosing which tokens get masked."""
    digest = hashlib.sha256(f"{base_seed}:{chembl_id}:{percent}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def generation_seed(base_seed: int, chembl_id: str, percent: float, temperature: float) -> int:
    """
    Per-(molecule, percent, temperature) seed for the ChemBERTa sampling draw.

    Reduced mod 2**63-1 to stay within the signed 64-bit range torch.manual_seed
    accepts (sha256 digest bits alone can exceed that range).
    """
    digest = hashlib.sha256(
        f"{base_seed}:{chembl_id}:{percent}:{temperature}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % (2 ** 63 - 1)


# ════════════════════════════════════════════════════════════════════════════
#  TOKEN-LEVEL RANDOM MASKING (D1)
# ════════════════════════════════════════════════════════════════════════════

def mask_smiles_tokens(
    smiles: str,
    percent: float,
    tokenizer,
    seed: int,
) -> Tuple[str, int, int]:
    """
    Mask `percent`% of the SMILES' BPE tokens directly (no RDKit atom mapping).

    Tokenizes the raw SMILES (no special tokens), samples that percent of
    token positions uniformly at random (min 1, capped at total tokens), and
    replaces each chosen token's character span with exactly one <mask>.

    Returns (masked_smiles, n_tokens, n_masked).
    """
    enc = tokenizer(smiles, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"] if int(e) > int(s)]
    n_tokens = len(offsets)
    if n_tokens == 0:
        return smiles, 0, 0

    n_mask = max(1, round(percent / 100.0 * n_tokens))
    n_mask = min(n_mask, n_tokens)

    rng = random.Random(seed)
    mask_positions = sorted(rng.sample(range(n_tokens), n_mask))
    spans = [offsets[i] for i in mask_positions]

    masked_smiles = apply_bpe_mask_spans(smiles, spans, tokenizer.mask_token)
    return masked_smiles, n_tokens, n_mask


# ════════════════════════════════════════════════════════════════════════════
#  SINGLE-SHOT GENERATION (D5)
# ════════════════════════════════════════════════════════════════════════════

def generate_one_molecule(
    masked_smiles: str,
    tokenizer,
    model,
    device: str,
    temperature: float,
    seed: int,
) -> Tuple[str, bool]:
    """
    Draw exactly one ChemBERTa candidate for masked_smiles, seeded for
    reproducibility. Returns (generated_smiles, is_rdkit_valid).

    generated_smiles is "" when the single draw was not RDKit-valid (the
    row is still written by the caller so every combo has exactly one
    record per input molecule).
    """
    torch.manual_seed(seed)
    candidates = generate_smiles(
        masked_smiles, tokenizer, model, device,
        top_k=config.TOP_K,
        num_samples=1,
        temperature=temperature,
    )
    if candidates:
        return candidates[0], True
    return "", False


# ════════════════════════════════════════════════════════════════════════════
#  PER-COMBO CSV I/O (D4, resumable)
# ════════════════════════════════════════════════════════════════════════════

def combo_csv_path(output_dir: str, percent: float, temperature: float, seed: int) -> str:
    fname = f"mask{percent:g}pct_temp{temperature:g}_seed{seed}.csv"
    return os.path.join(output_dir, fname)


def load_done_ids(csv_path: str) -> set:
    done: set = set()
    if os.path.isfile(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["chembl_id"])
    return done


# ════════════════════════════════════════════════════════════════════════════
#  MAIN SWEEP
# ════════════════════════════════════════════════════════════════════════════

def _process_one_molecule(
    chembl_id: str,
    smiles: str,
    percent: float,
    temperature: float,
    seed: int,
    tokenizer,
    model,
    device: str,
    rdkit_log_f,
) -> Optional[dict]:
    """
    Mask + generate for one molecule, with all RDKit stdout/stderr chatter
    (e.g. "SMILES Parse Error") captured into an in-memory buffer instead of
    the console. If anything was captured, it is appended to rdkit_log_f
    tagged with the (chembl_id, percent, temperature, seed) that produced it.

    Returns the CSV row dict, or None if the input SMILES itself was invalid
    (should not happen — Stage 0a already RDKit-verified it).
    """
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        mol_ok = Chem.MolFromSmiles(smiles) is not None
        row = None
        if mol_ok:
            mseed = mask_seed(seed, chembl_id, percent)
            masked_smiles, n_tokens, n_masked = mask_smiles_tokens(
                smiles, percent, tokenizer, mseed
            )

            gseed = generation_seed(seed, chembl_id, percent, temperature)
            generated_smiles, valid = generate_one_molecule(
                masked_smiles, tokenizer, model, device, temperature, gseed
            )

            row = {
                "chembl_id":        chembl_id,
                "smiles":           smiles,
                "mask_percent":     percent,
                "temperature":      temperature,
                "seed":             seed,
                "n_tokens":         n_tokens,
                "n_masked":         n_masked,
                "masked_smiles":    masked_smiles,
                "generated_smiles": generated_smiles,
                "valid":            valid,
            }

    captured = buf.getvalue()
    if captured:
        rdkit_log_f.write(
            f"[{chembl_id} | pct={percent} T={temperature} seed={seed}]\n{captured}\n"
        )
        rdkit_log_f.flush()

    return row


def run_stage1a(
    molecules: Optional[List[Tuple[str, str]]] = None,
    percents: Optional[List[float]] = None,
    temperatures: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run the full (percent × temperature × seed) sweep over `molecules`,
    writing one resumable CSV per combo to output_dir.

    RDKit's parse-error/warning chatter is captured per molecule and appended
    to <output_dir>/rdkit_errors.txt (tagged by molecule/combo) instead of
    interleaving with the tqdm progress bar on the console.
    """
    output_dir   = output_dir or config.STAGE1A_DIR
    percents     = percents or config.STAGE1A_MASK_PERCENTS
    temperatures = temperatures or config.STAGE1A_TEMPERATURES
    seeds        = seeds or config.RANDOM_MASK_SEEDS_LIST
    os.makedirs(output_dir, exist_ok=True)

    if molecules is None:
        molecules = load_stage0a_smiles(limit=getattr(config, "STAGE1A_INPUT_LIMIT", None))
    if not molecules:
        print("  No Stage-0a molecules available. Exiting.")
        return

    tokenizer, model, device = load_chemberta()

    total_combos = len(percents) * len(temperatures) * len(seeds)
    combo_i = 0
    rdkit_log_path = os.path.join(output_dir, "rdkit_errors.txt")

    with open(rdkit_log_path, "a", encoding="utf-8") as rdkit_log_f:
        for percent in percents:
            for temperature in temperatures:
                for seed in seeds:
                    combo_i += 1
                    csv_path = combo_csv_path(output_dir, percent, temperature, seed)
                    done_ids = load_done_ids(csv_path)
                    write_header = not os.path.isfile(csv_path)

                    desc = f"[{combo_i}/{total_combos}] pct={percent} T={temperature} seed={seed}"
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                        if write_header:
                            writer.writeheader()

                        for chembl_id, smiles in tqdm(molecules, desc=desc, unit="mol"):
                            if chembl_id in done_ids:
                                continue

                            row = _process_one_molecule(
                                chembl_id, smiles, percent, temperature, seed,
                                tokenizer, model, device, rdkit_log_f,
                            )
                            if row is None:
                                continue  # invalid input SMILES (logged above)

                            writer.writerow(row)
                            f.flush()


# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY METRIC (D6): unique-valid / total, averaged across seeds
# ════════════════════════════════════════════════════════════════════════════

def _canonical_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compute_combo_metric(csv_path: str) -> Tuple[float, int]:
    """
    Read one (percent, temperature, seed) combo CSV.

    Returns (unique_valid_ratio, total_molecules), where
      unique_valid_ratio = (# distinct RDKit-canonical valid generated
                             molecules) / (# molecules attempted in this combo)
    """
    total = 0
    canonical_set: set = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            if row.get("valid") == "True" and row.get("generated_smiles"):
                canon = _canonical_smiles(row["generated_smiles"])
                if canon:
                    canonical_set.add(canon)

    if total == 0:
        return 0.0, 0
    return len(canonical_set) / total, total


def aggregate_stage1a_results(
    percents: Optional[List[float]] = None,
    temperatures: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> Tuple[Dict[float, Dict[float, float]], int]:
    """
    Build {temperature: {percent: mean_unique_valid_ratio_across_seeds}}.

    A missing combo CSV (e.g. a partial sweep) is skipped with a warning so
    the remaining combos can still be plotted. Returns the results dict plus
    the max molecule count seen across combos (for the plot subtitle).
    """
    output_dir   = output_dir or config.STAGE1A_DIR
    percents     = percents or config.STAGE1A_MASK_PERCENTS
    temperatures = temperatures or config.STAGE1A_TEMPERATURES
    seeds        = seeds or config.RANDOM_MASK_SEEDS_LIST

    results: Dict[float, Dict[float, float]] = {}
    total_molecules = 0

    for temperature in temperatures:
        results[temperature] = {}
        for percent in percents:
            seed_ratios: List[float] = []
            for seed in seeds:
                csv_path = combo_csv_path(output_dir, percent, temperature, seed)
                if not os.path.isfile(csv_path):
                    print(f"  ⚠️  Missing combo CSV, skipped in plot: {csv_path}")
                    continue
                ratio, total = compute_combo_metric(csv_path)
                seed_ratios.append(ratio)
                total_molecules = max(total_molecules, total)

            if seed_ratios:
                results[temperature][percent] = sum(seed_ratios) / len(seed_ratios)

    return results, total_molecules


def write_stage1a_summary_csv(
    results: Dict[float, Dict[float, float]],
    output_path: str,
) -> str:
    """Write the aggregated per-(temperature, percent) means to a CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["temperature", "mask_percent", "mean_unique_valid_ratio"])
        for temperature in sorted(results):
            for percent in sorted(results[temperature]):
                writer.writerow([temperature, percent, results[temperature][percent]])
    return output_path


# ════════════════════════════════════════════════════════════════════════════
#  PLOT: unique-valid ratio vs mask percent, one curve per temperature
# ════════════════════════════════════════════════════════════════════════════

# Fixed categorical color order (never reassigned based on which temperatures
# are present) — validated CVD-safe order, see dataviz skill references/palette.md.
_STAGE1A_PALETTE = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]


def plot_stage1a_validity(
    results: Dict[float, Dict[float, float]],
    total_molecules: int,
    seeds: List[int],
    plot_path: str,
) -> str:
    """
    x-axis : mask percent
    y-axis : unique valid molecules / total molecules (mean across seeds)
    One line per temperature, colors assigned in fixed order.
    """
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)

    surface   = "#fcfcfb"
    gridline  = "#e1e0d9"
    axis_ink  = "#898781"
    label_ink = "#52514e"
    title_ink = "#0b0b0b"

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    all_percents: set = set()
    for temperature in sorted(results):
        per_percent = results[temperature]
        xs = sorted(per_percent)
        if not xs:
            continue
        ys = [per_percent[p] for p in xs]
        all_percents.update(xs)

        color = _STAGE1A_PALETTE[sorted(results).index(temperature) % len(_STAGE1A_PALETTE)]
        ax.plot(
            xs, ys,
            marker="o", markersize=8, linewidth=2,
            color=color, label=f"T = {temperature:g}",
        )

    ax.set_xlabel("Tokens masked (%)", fontsize=12, color=label_ink)
    ax.set_ylabel(
        "Unique valid molecules / total molecules\n(mean across seeds)",
        fontsize=12, color=label_ink,
    )
    ax.set_title(
        f"Stage 1a: ChemBERTa single-shot generation yield\n"
        f"(n = {total_molecules} molecules, seeds = {seeds})",
        fontsize=13, color=title_ink,
    )
    if all_percents:
        ax.set_xticks(sorted(all_percents))
    ax.tick_params(colors=axis_ink)
    ax.grid(True, color=gridline, linewidth=1)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.legend(fontsize=11, frameon=False)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")
    return plot_path


def summarize_and_plot_stage1a(
    percents: Optional[List[float]] = None,
    temperatures: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Aggregate combo CSVs in output_dir and save the validity plot + summary CSV."""
    output_dir   = output_dir or config.STAGE1A_DIR
    percents     = percents or config.STAGE1A_MASK_PERCENTS
    temperatures = temperatures or config.STAGE1A_TEMPERATURES
    seeds        = seeds or config.RANDOM_MASK_SEEDS_LIST

    results, total_molecules = aggregate_stage1a_results(
        percents=percents, temperatures=temperatures, seeds=seeds, output_dir=output_dir,
    )

    write_stage1a_summary_csv(
        results, os.path.join(output_dir, "stage1a_validity_summary.csv")
    )
    return plot_stage1a_validity(
        results,
        total_molecules,
        seeds,
        os.path.join(output_dir, "stage1a_validity_vs_mask_percent.png"),
    )


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 60)
    print("STAGE 1a: CHEMBL RANDOM TOKEN MASKING + SINGLE-SHOT GENERATION")
    print("=" * 60)
    print(f"""
  Reads verified ChEMBL SMILES from:
    {config.STAGE0A_DIR}
  Mask percents : {config.STAGE1A_MASK_PERCENTS}
  Temperatures  : {config.STAGE1A_TEMPERATURES}
  Seeds         : {config.RANDOM_MASK_SEEDS_LIST}
  Input limit   : {getattr(config, 'STAGE1A_INPUT_LIMIT', None)}

  Writes one CSV per (percent, temperature, seed) combo to:
    {config.STAGE1A_DIR}
  RDKit parse errors/warnings (kept off the progress bar) are dumped to:
    {os.path.join(config.STAGE1A_DIR, 'rdkit_errors.txt')}
""")

    run_stage1a()
    summarize_and_plot_stage1a()

    print(f"\n  Stage 1a complete. Output: {config.STAGE1A_DIR}")


def _run_self_test() -> None:
    """Quick smoke test: 2 synthetic molecules, 2 percents, 2 temperatures, 2 seeds."""
    import tempfile

    test_molecules = [
        ("TEST_ASPIRIN", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("TEST_CAFFEINE", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ]
    percents     = [10, 20]
    temperatures = [0.5, 1.2]
    seeds        = [17, 53]

    with tempfile.TemporaryDirectory() as td:
        run_stage1a(
            molecules=test_molecules,
            percents=percents,
            temperatures=temperatures,
            seeds=seeds,
            output_dir=td,
        )
        csv_path = combo_csv_path(td, percents[0], temperatures[0], seeds[0])
        assert os.path.isfile(csv_path), f"Expected output CSV at {csv_path}"

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(test_molecules), (
            f"Expected {len(test_molecules)} rows, got {len(rows)}"
        )
        for row in rows:
            assert row["masked_smiles"] != row["smiles"], "Masking had no effect"
            assert int(row["n_masked"]) >= 1, "Expected at least one masked token"

        results, total_molecules = aggregate_stage1a_results(
            percents=percents, temperatures=temperatures, seeds=seeds, output_dir=td,
        )
        assert total_molecules == len(test_molecules), (
            f"Expected total_molecules={len(test_molecules)}, got {total_molecules}"
        )
        assert set(results.keys()) == set(temperatures), "Missing a temperature curve"
        for temperature in temperatures:
            assert set(results[temperature].keys()) == set(percents), (
                f"Missing a percent point for T={temperature}"
            )
            for ratio in results[temperature].values():
                assert 0.0 <= ratio <= 1.0, f"Ratio out of range: {ratio}"

        plot_path = summarize_and_plot_stage1a(
            percents=percents, temperatures=temperatures, seeds=seeds, output_dir=td,
        )
        assert os.path.isfile(plot_path), f"Expected plot at {plot_path}"
        summary_csv = os.path.join(td, "stage1a_validity_summary.csv")
        assert os.path.isfile(summary_csv), f"Expected summary CSV at {summary_csv}"
        with open(summary_csv, newline="", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))
        assert len(summary_rows) == len(percents) * len(temperatures), (
            f"Expected {len(percents) * len(temperatures)} summary rows, "
            f"got {len(summary_rows)}"
        )

    print("✅ Stage 1a self-test passed.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main()

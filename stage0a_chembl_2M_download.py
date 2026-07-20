# -*- coding: utf-8 -*-
"""
stage0a_chembl_2M_download.py
==============================
Download ChEMBL (latest release) chemical representations, filter SMILES
with RDKit for validity, and save a verified SMILES file for use in later
pipeline stages.

Source file: chembl_XX_chemreps.txt.gz from the EBI FTP server.
  Columns: chembl_id | canonical_smiles | standard_inchi | standard_inchi_key

Output:
  <CHEMBL_OUT_DIR>/chembl_verified_smiles.csv  (chembl_id, smiles)
  <CHEMBL_OUT_DIR>/chembl_download_stats.txt   (counts, rejections)

Usage:
  python stage0a_chembl_2M_download.py

Or on a Colab/Linux machine where wget is faster:
  wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_chemreps.txt.gz
  Then set CHEMREPS_LOCAL below to that path and re-run (download is skipped).
"""

import gzip
import os
import time
import urllib.request
from typing import Optional

import config

# ── Configuration ─────────────────────────────────────────────────────────────

# EBI FTP — update version number if a newer ChEMBL release is available
CHEMBL_VERSION   = 37
CHEMREPS_URL     = (
    f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
    f"chembl_{CHEMBL_VERSION}_chemreps.txt.gz"
)

# Output directory — dedicated Stage 0a directory from config
CHEMBL_OUT_DIR   = config.STAGE0A_DIR

# If you already downloaded the .gz file manually, set this path to skip download.
# Leave as None to download automatically.
CHEMREPS_LOCAL: Optional[str] = None

# ── Download helper ────────────────────────────────────────────────────────────

def _download_with_progress(url: str, dest: str) -> None:
    """Stream-download url to dest, printing MB progress."""
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    def _hook(count, block_size, total_size):
        done = count * block_size
        if total_size > 0:
            pct = min(100.0, done / total_size * 100)
            mb_done  = done / 1_048_576
            mb_total = total_size / 1_048_576
            print(f"\r  {pct:5.1f}%  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()   # newline after progress bar


# ── Main ──────────────────────────────────────────────────────────────────────

def run_stage0a(
    chemreps_local: Optional[str] = None,
    out_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Download (if needed), parse, RDKit-validate, and save ChEMBL SMILES.

    Parameters
    ----------
    chemreps_local : path to a pre-downloaded chembl_XX_chemreps.txt.gz,
                     or None to download automatically.
    out_dir        : output directory (defaults to CHEMBL_OUT_DIR).
    limit          : optional cap on molecules processed (for quick testing).

    Returns
    -------
    Path to the saved CSV file.
    """
    # Late import so the script can be imported without rdkit installed
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit is required: pip install rdkit")

    import csv

    out_dir = out_dir or CHEMBL_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    gz_path = chemreps_local or CHEMREPS_LOCAL
    if not gz_path:
        gz_path = os.path.join(out_dir, f"chembl_{CHEMBL_VERSION}_chemreps.txt.gz")

    if os.path.exists(gz_path):
        print(f"  Found existing file, skipping download: {gz_path}")
    else:
        _download_with_progress(CHEMREPS_URL, gz_path)

    out_csv  = os.path.join(out_dir, "chembl_verified_smiles.csv")
    out_stats = os.path.join(out_dir, "chembl_download_stats.txt")

    print(f"\n  Parsing and RDKit-validating: {gz_path}")
    t0 = time.time()

    total     = 0
    valid     = 0
    invalid   = 0
    no_smiles = 0

    with gzip.open(gz_path, "rt", encoding="utf-8") as f_in, \
         open(out_csv, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(["chembl_id", "smiles"])

        header = f_in.readline()   # skip header line
        col_names = [c.strip() for c in header.split("\t")]
        try:
            idx_id     = col_names.index("chembl_id")
            idx_smiles = col_names.index("canonical_smiles")
        except ValueError:
            # Fall back: assume first two columns
            idx_id, idx_smiles = 0, 1

        for line in f_in:
            if not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= max(idx_id, idx_smiles):
                continue

            total += 1
            chembl_id = cols[idx_id].strip()
            smi       = cols[idx_smiles].strip()

            if not smi or smi.lower() in ("none", "null", ""):
                no_smiles += 1
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid += 1
                continue

            canonical = Chem.MolToSmiles(mol)
            writer.writerow([chembl_id, canonical])
            valid += 1

            if valid % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"    {valid:>7,} valid  |  {total:>7,} processed  |  {elapsed:.0f}s")

            if limit and total >= limit:
                print(f"  [limit={limit} reached]")
                break

    elapsed = time.time() - t0
    reject_rate = (invalid + no_smiles) / max(total, 1) * 100

    stats_lines = [
        f"ChEMBL {CHEMBL_VERSION} download stats",
        f"  Source              : {gz_path}",
        f"  Total rows parsed   : {total:,}",
        f"  Valid (RDKit)       : {valid:,}",
        f"  Invalid SMILES      : {invalid:,}",
        f"  Missing SMILES      : {no_smiles:,}",
        f"  Rejection rate      : {reject_rate:.2f}%",
        f"  Elapsed             : {elapsed:.1f}s",
        f"  Output CSV          : {out_csv}",
    ]
    stats_text = "\n".join(stats_lines)
    print("\n" + stats_text)

    with open(out_stats, "w", encoding="utf-8") as f:
        f.write(stats_text + "\n")

    return out_csv


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = run_stage0a(
        chemreps_local = CHEMREPS_LOCAL,
        out_dir        = CHEMBL_OUT_DIR,
    )
    print(f"\n  Done. Verified SMILES saved to: {csv_path}")

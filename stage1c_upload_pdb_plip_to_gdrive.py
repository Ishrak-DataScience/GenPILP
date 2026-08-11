# -*- coding: utf-8 -*-
"""
stage1c_upload_pdb_plip_to_gdrive.py
=====================================
Uploads local PDB / PLIP-XML file PAIRS to Google Drive.

Reuses Stage 1b's pair-discovery logic unmodified:
    Local PDB mirror   :  <pdb_root>/<mid2>/pdb<id>.ent.gz  (mid2 = id[1:3])
    Pre-computed PLIP   :  <xml_root>/pdb<id>.xml            (flat)

Only pdb_ids present in BOTH locations are uploaded — a pdb_id missing its
PDB structure or its PLIP XML is skipped entirely (never uploaded alone).

Two upload modes (config-driven, both overridable via CLI):
    "all" : upload every eligible pair
    "n"   : upload a random sample of N eligible pairs (seeded, reproducible)

Resume/checkpoint (same pattern as Stage 1b D7): every successfully-uploaded
pair is appended to a manifest CSV, flushed + fsync'd immediately. Re-running
the script skips pdb_ids already recorded in the manifest.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED LIBRARIES (Linux server -> Google Drive upload)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pip install --user google-api-python-client google-auth-oauthlib google-auth-httplib2 tqdm

(gdown is NOT used here — gdown only reliably supports *downloading* from
Drive; its upload support is unofficial/unmaintained. Uploads use Google's
official Drive API v3 client instead.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTHENTICATION — pick ONE (both configured in config.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A) Service account (recommended for a headless Linux server — no browser):
   1. Google Cloud Console -> IAM & Admin -> Service Accounts -> create one.
   2. Keys -> Add key -> JSON; save it on the server, point
      config.GDRIVE_SERVICE_ACCOUNT_FILE at it.
   3. Share the target Drive folder with the service account's client_email
      (found inside the JSON key). Service accounts have no personal storage
      quota, so the target folder should live on a Shared Drive, not "My
      Drive" of a regular user.
   4. Set config.GDRIVE_FOLDER_ID to that folder's id.

B) OAuth "installed app" flow (uploads land in YOUR own My Drive):
   1. Google Cloud Console -> APIs & Services -> Credentials -> Create
      credentials -> OAuth client ID -> Desktop app -> download JSON, point
      config.GDRIVE_CREDENTIALS_FILE at it.
   2. First run needs a one-time browser authorization. On a headless
      server, either forward the local-server port over SSH
      (ssh -L 8080:localhost:8080 user@server) and open the printed URL
      on your own machine, OR run this script once on a machine with a
      browser to generate config.GDRIVE_TOKEN_FILE, then copy that token
      file to the server. Subsequent runs reuse the cached token
      (auto-refreshed) with no browser involved.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Standalone (interactive prompts for anything not passed):
        python stage1c_upload_pdb_plip_to_gdrive.py

    Batch, fixed-size random sample of 100 pairs (no prompts):
        python stage1c_upload_pdb_plip_to_gdrive.py \\
            --upload-mode n --n 100 --seed 42 \\
            --pdb-root /group/bioinf_tmp/Data/pdb \\
            --xml-root /group/bioinf_tmp/plip_pdb2xml \\
            --folder-id <drive_folder_id>

    Batch, ALL eligible pairs (safe to re-run after a crash — pdb_ids
    already in the manifest are skipped automatically):
        python stage1c_upload_pdb_plip_to_gdrive.py \\
            --upload-mode all \\
            --pdb-root /group/bioinf_tmp/Data/pdb \\
            --xml-root /group/bioinf_tmp/plip_pdb2xml \\
            --folder-id <drive_folder_id>

    Preview what would be uploaded without calling the Drive API or needing
    credentials at all:
        python stage1c_upload_pdb_plip_to_gdrive.py --upload-mode n --n 20 --dry-run

HOW TO TEST (no Drive credentials needed — Drive calls are monkeypatched out)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python stage1c_upload_pdb_plip_to_gdrive.py --test
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List, Optional

import config
from stage1b_large_scale_PLIP_mask_calculation import (
    discover_available_ids,
    sample_ids,
    _pdb_gz_path,
    _load_completed_ids,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else []


MANIFEST_FIELDS = [
    "pdb_id", "status", "error",
    "pdb_filename", "xml_filename",
    "pdb_file_id", "xml_file_id",
]

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


# ════════════════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE AUTH + UPLOAD
# ════════════════════════════════════════════════════════════════════════════

def get_drive_service(
    service_account_file: str,
    credentials_file: str,
    token_file: str,
):
    """
    Build an authenticated Drive v3 service. Service-account auth (no
    browser) is used if service_account_file is set and exists; otherwise
    falls back to the OAuth installed-app flow, caching the token so only
    the very first run needs browser interaction (see module docstring).
    """
    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise ImportError(
            "Missing Google API client libraries. Install with:\n"
            "  pip install --user google-api-python-client google-auth-oauthlib google-auth-httplib2"
        ) from e

    if service_account_file and os.path.isfile(service_account_file):
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=SCOPES
        )
        return build("drive", "v3", credentials=creds)

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.isfile(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.isfile(credentials_file):
                raise FileNotFoundError(
                    f"No Drive credentials found. Either set "
                    f"config.GDRIVE_SERVICE_ACCOUNT_FILE to a service-account "
                    f"JSON key, or set config.GDRIVE_CREDENTIALS_FILE to an "
                    f"OAuth 'Desktop app' client-secret JSON (missing: "
                    f"{credentials_file})."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        os.makedirs(os.path.dirname(token_file) or ".", exist_ok=True)
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


def upload_file(service, local_path: str, folder_id: str, mimetype: str) -> str:
    """Resumable upload of one local file to Drive; returns the new file's id."""
    from googleapiclient.http import MediaFileUpload

    metadata = {"name": os.path.basename(local_path)}
    if folder_id:
        metadata["parents"] = [folder_id]
    media = MediaFileUpload(local_path, mimetype=mimetype, resumable=True, chunksize=10 * 1024 * 1024)
    request = service.files().create(body=metadata, media_body=media, fields="id")
    response = None
    while response is None:
        _status, response = request.next_chunk()
    return response["id"]


def upload_file_with_retry(
    service, local_path: str, folder_id: str, mimetype: str,
    max_attempts: int = 5,
) -> str:
    """upload_file() with exponential backoff on transient Drive/network errors."""
    from googleapiclient.errors import HttpError

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return upload_file(service, local_path, folder_id, mimetype)
        except HttpError as e:
            last_err = e
            if e.resp is not None and e.resp.status not in (403, 429, 500, 502, 503, 504):
                raise
        except Exception as e:  # network hiccups, etc.
            last_err = e
        if attempt < max_attempts:
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Upload failed after {max_attempts} attempts: {last_err}") from last_err


# ════════════════════════════════════════════════════════════════════════════
#  BATCH DRIVER
# ════════════════════════════════════════════════════════════════════════════

def upload_pdb_plip_pairs(
    pdb_root: str,
    xml_root: str,
    upload_mode: str,
    n: int,
    seed: int,
    folder_id: str,
    manifest_path: str,
    resume: bool,
    service_account_file: str,
    credentials_file: str,
    token_file: str,
    dry_run: bool = False,
) -> str:
    """
    Select eligible pdb_ids (D2-style: "all" = every pair present in both
    roots, "n" = a random sample of size n), upload each pair's two files,
    and append one row per pdb_id to the manifest CSV as it completes
    (flushed + fsync'd, so an interrupted run can resume). Returns the
    manifest CSV path.
    """
    available = discover_available_ids(pdb_root, xml_root)
    if not available:
        raise RuntimeError(f"No PDB/XML pairs found (pdb_root={pdb_root}, xml_root={xml_root}).")

    if upload_mode == "all":
        picked = list(available)
        print(f"  ALL mode: {len(available)} PDB/XML pairs available.")
    else:
        picked = sample_ids(available, n, seed)
        print(f"  {len(available)} PDB/XML pairs available; sampled {len(picked)} (seed={seed}).")

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    completed = set()
    if resume:
        completed = _load_completed_ids(manifest_path)
        if completed:
            before = len(picked)
            picked = [pid for pid in picked if pid not in completed]
            print(f"  Resume: {before - len(picked)} of {before} already in {manifest_path} — skipping them.")

    if not picked:
        print("  Nothing left to upload (all picked pairs already uploaded).")
        if not (resume and os.path.isfile(manifest_path) and completed):
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()
        return manifest_path

    service = None if dry_run else get_drive_service(service_account_file, credentials_file, token_file)

    file_exists_with_header = resume and os.path.isfile(manifest_path) and bool(completed)
    file_mode = "a" if file_exists_with_header else "w"
    with open(manifest_path, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if file_mode == "w":
            writer.writeheader()

        for pdb_id in tqdm(picked, desc="Uploading PDB/PLIP pairs", unit="pair"):
            pdb_path = _pdb_gz_path(pdb_root, pdb_id)
            xml_path = os.path.join(xml_root, f"pdb{pdb_id}.xml")
            row = {
                "pdb_id": pdb_id, "status": "ok", "error": "",
                "pdb_filename": os.path.basename(pdb_path),
                "xml_filename": os.path.basename(xml_path),
                "pdb_file_id": "", "xml_file_id": "",
            }
            try:
                if dry_run:
                    row["pdb_file_id"] = "DRY_RUN"
                    row["xml_file_id"] = "DRY_RUN"
                else:
                    row["pdb_file_id"] = upload_file_with_retry(
                        service, pdb_path, folder_id, "application/gzip"
                    )
                    row["xml_file_id"] = upload_file_with_retry(
                        service, xml_path, folder_id, "application/xml"
                    )
            except Exception as e:
                row["status"] = "error"
                row["error"] = str(e)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    with open(manifest_path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    n_ok = sum(1 for r in all_rows if r["status"] == "ok")
    print(f"  {n_ok}/{len(all_rows)} pairs uploaded successfully (cumulative across resumes).")
    print(f"  Manifest CSV: {manifest_path}")
    return manifest_path


# ════════════════════════════════════════════════════════════════════════════
#  CLI / INTERACTIVE PROMPTS
# ════════════════════════════════════════════════════════════════════════════

def _prompt_str(question: str, default: str) -> str:
    raw = input(f"  {question} [{default}]: ").strip()
    return raw or default


def _prompt_int(question: str, default: int) -> int:
    while True:
        raw = input(f"  {question} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("    Please enter an integer.")


def _ask_upload_mode() -> str:
    print("""
  Upload mode:
    1 : Fixed sample size N, drawn at random             [default]
    2 : ALL eligible PDB/XML pairs
""")
    while True:
        raw = input("  Select upload mode (1 / 2) [1]: ").strip()
        if raw in ("", "1"):
            return "n"
        if raw == "2":
            return "all"
        print("    Please type 1 or 2.")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload matching local PDB / PLIP-XML file pairs to Google Drive."
    )
    p.add_argument("--upload-mode", type=str, choices=["n", "all"], default=None,
                    help="'n' = upload --n random pairs (default); 'all' = upload every eligible pair.")
    p.add_argument("--n", type=int, default=None, help="Number of pairs to upload (--upload-mode n).")
    p.add_argument("--seed", type=int, default=None, help="Random sample seed (--upload-mode n).")
    p.add_argument("--pdb-root", type=str, default=None, help="Root of the local PDB mirror.")
    p.add_argument("--xml-root", type=str, default=None, help="Directory of pre-computed PLIP XML files.")
    p.add_argument("--folder-id", type=str, default=None, help="Target Google Drive folder id.")
    p.add_argument("--service-account-file", type=str, default=None,
                    help="Service-account JSON key (headless auth, no browser).")
    p.add_argument("--credentials-file", type=str, default=None,
                    help="OAuth 'Desktop app' client-secret JSON (used if no service account).")
    p.add_argument("--token-file", type=str, default=None, help="Cached OAuth token path.")
    p.add_argument("--manifest", type=str, default=None, help="Upload-manifest CSV path.")
    p.add_argument("--no-resume", action="store_true",
                    help="Disable resume — by default, pdb_ids already recorded in the manifest are skipped.")
    p.add_argument("--dry-run", action="store_true",
                    help="Select pairs and write the manifest without calling the Drive API or needing credentials.")
    p.add_argument("--test", action="store_true", help="Run the self-test and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    print("\n" + "=" * 60)
    print("STAGE 1c: UPLOAD PDB/PLIP PAIRS TO GOOGLE DRIVE")
    print("=" * 60)

    args = _parse_args(argv)

    upload_mode = args.upload_mode or getattr(config, "GDRIVE_UPLOAD_MODE", None) or _ask_upload_mode()
    if upload_mode == "n":
        n = args.n if args.n is not None else _prompt_int(
            "Number of pairs to upload (N)", getattr(config, "GDRIVE_UPLOAD_N", 50)
        )
        seed = args.seed if args.seed is not None else _prompt_int(
            "Random sample seed", getattr(config, "GDRIVE_UPLOAD_SEED", 42)
        )
    else:
        n = args.n if args.n is not None else 0
        seed = args.seed if args.seed is not None else getattr(config, "GDRIVE_UPLOAD_SEED", 42)

    resume = getattr(config, "GDRIVE_UPLOAD_RESUME", True) and not args.no_resume
    pdb_root = args.pdb_root or _prompt_str("PDB mirror root", config.PLIP_LARGE_SCALE_PDB_ROOT)
    xml_root = args.xml_root or _prompt_str("PLIP XML root", config.PLIP_LARGE_SCALE_XML_ROOT)
    folder_id = args.folder_id if args.folder_id is not None else getattr(config, "GDRIVE_FOLDER_ID", "")
    service_account_file = args.service_account_file or getattr(config, "GDRIVE_SERVICE_ACCOUNT_FILE", "")
    credentials_file = args.credentials_file or getattr(config, "GDRIVE_CREDENTIALS_FILE", "")
    token_file = args.token_file or getattr(config, "GDRIVE_TOKEN_FILE", "")
    manifest_path = args.manifest or getattr(config, "GDRIVE_UPLOAD_MANIFEST", "upload_manifest.csv")

    print(f"""
  Upload mode      : {"ALL eligible pairs" if upload_mode == "all" else f"N={n} (seed={seed})"}
  PDB root         : {pdb_root}
  XML root         : {xml_root}
  Drive folder id  : {folder_id or "(My Drive root)"}
  Manifest CSV     : {manifest_path}
  Resume           : {resume}
  Dry run          : {args.dry_run}
""")

    upload_pdb_plip_pairs(
        pdb_root=pdb_root, xml_root=xml_root, upload_mode=upload_mode,
        n=n, seed=seed, folder_id=folder_id, manifest_path=manifest_path,
        resume=resume, service_account_file=service_account_file,
        credentials_file=credentials_file, token_file=token_file,
        dry_run=args.dry_run,
    )

    print("\n✅ Stage 1c Google Drive upload complete.")


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST (no Drive credentials or network access required)
# ════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    import gzip
    import tempfile

    with tempfile.TemporaryDirectory() as pdb_root, tempfile.TemporaryDirectory() as xml_root, \
            tempfile.TemporaryDirectory() as out_dir:
        os.makedirs(os.path.join(pdb_root, "00"), exist_ok=True)
        with gzip.open(_pdb_gz_path(pdb_root, "100d"), "wb") as f:
            f.write(b"HEADER    fake pdb fixture\n")
        with open(os.path.join(xml_root, "pdb100d.xml"), "w", encoding="utf-8") as f:
            f.write("<report/>")
        # id "200e" has only an XML file (no PDB) -> must be excluded as a pair.
        with open(os.path.join(xml_root, "pdb200e.xml"), "w", encoding="utf-8") as f:
            f.write("<report/>")

        manifest_path = os.path.join(out_dir, "upload_manifest.csv")
        result_path = upload_pdb_plip_pairs(
            pdb_root=pdb_root, xml_root=xml_root, upload_mode="all",
            n=0, seed=0, folder_id="", manifest_path=manifest_path, resume=True,
            service_account_file="", credentials_file="", token_file="",
            dry_run=True,
        )
        assert result_path == manifest_path
        with open(manifest_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1, f"Expected exactly 1 eligible pair (100d only), got {rows}"
        assert rows[0]["pdb_id"] == "100d"
        assert rows[0]["status"] == "ok"
        assert rows[0]["pdb_file_id"] == "DRY_RUN"

        # Re-running with resume=True must skip the already-uploaded id.
        upload_pdb_plip_pairs(
            pdb_root=pdb_root, xml_root=xml_root, upload_mode="all",
            n=0, seed=0, folder_id="", manifest_path=manifest_path, resume=True,
            service_account_file="", credentials_file="", token_file="",
            dry_run=True,
        )
        with open(manifest_path, newline="", encoding="utf-8") as f:
            rows_resumed = list(csv.DictReader(f))
        assert rows_resumed == rows, "Resume should leave already-uploaded rows untouched"

        # "n" mode with n=1 against a single eligible pair must pick it.
        manifest_path_n = os.path.join(out_dir, "upload_manifest_n.csv")
        upload_pdb_plip_pairs(
            pdb_root=pdb_root, xml_root=xml_root, upload_mode="n",
            n=1, seed=0, folder_id="", manifest_path=manifest_path_n, resume=True,
            service_account_file="", credentials_file="", token_file="",
            dry_run=True,
        )
        with open(manifest_path_n, newline="", encoding="utf-8") as f:
            rows_n = list(csv.DictReader(f))
        assert len(rows_n) == 1 and rows_n[0]["pdb_id"] == "100d"

    print("✅ Stage 1c Google Drive upload self-test passed.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        main()

"""Data loading and Kaggle download."""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

# Kaggle dataset: Szymon Jóźwiak, "NBA Play-by-Play Data 1997–2023"
KAGGLE_DATASET = "szymonjwiak/nba-play-by-play-data-1997-2023"


def download_kaggle_data(target_dir: Path) -> Path:
    """Download the NBA PbP dataset via Kaggle CLI and mirror to *target_dir*.

    Uses the official ``kaggle`` CLI (kaggle datasets download), which is more
    robust than the kagglehub library across Python versions.

    Returns the path to the directory containing ``pbp*.csv``.
    If *target_dir* already contains CSV files, skips the download.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = list(target_dir.glob("pbp*.csv"))
    if existing:
        print(f"[download] {len(existing)} season files already in {target_dir}, skipping.")
        return target_dir

    if shutil.which("kaggle"):
        print("[download] Using Kaggle CLI …")
        zip_path = target_dir / "nba_pbp.zip"
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", str(target_dir), "--unzip"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            csv_count = len(list(target_dir.glob("pbp*.csv")))
            if csv_count > 0:
                print(f"[download] {csv_count} season files downloaded to {target_dir}")
                return target_dir
            else:
                print("[download] Kaggle CLI succeeded but no CSV found. Checking zip …")
                if zip_path.exists():
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(target_dir)
                    zip_path.unlink()
        else:
            print(f"[download] Kaggle CLI failed: {result.stderr.strip()}")
    else:
        print("[download] Kaggle CLI not found (pip install kaggle).")

    try:
        import kagglehub
        print("[download] Falling back to kagglehub …")
        kagglehub.dataset_download(KAGGLE_DATASET)
    except ImportError:
        pass
    except Exception as e:
        print(f"[download] kagglehub also failed: {e}")

    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / \
                 "szymonjwiak" / "nba-play-by-play-data-1997-2023" / "versions"
    if cache_root.exists():
        version_dirs = sorted(cache_root.iterdir(), key=os.path.getmtime, reverse=True)
        for src_dir in version_dirs:
            for f in src_dir.rglob("pbp*.csv"):
                dest = target_dir / f.name
                if not dest.exists():
                    dest.write_bytes(f.read_bytes())
            csv_files = list(target_dir.glob("pbp*.csv"))
            if csv_files:
                print(f"[download] {len(csv_files)} season files recovered from cache.")
                return target_dir

    final_count = len(list(target_dir.glob("pbp*.csv")))
    if final_count == 0:
        raise FileNotFoundError(
            "No pbp*.csv files found.\n\n"
            "  Option A (recommended): pip install kaggle && kaggle datasets download "
            "-d szymonjwiak/nba-play-by-play-data-1997-2023 -p ./nba_pbp_data --unzip\n"
            "  Option B: pip install 'kagglehub<1.0'\n"
            "  Option C: scp pbp*.csv files to ./nba_pbp_data/ and use --skip-download"
        )
    print(f"[download] {final_count} season files available in {target_dir}")
    return target_dir

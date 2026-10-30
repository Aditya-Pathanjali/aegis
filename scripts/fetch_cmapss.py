"""Downloads NASA C-MAPSS turbofan degradation dataset and extracts FD001 sample."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
# Note: The SHA-256 below is the hash of the file downloaded from data.nasa.gov, not one published by NASA.
EXPECTED_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"

COLUMN_NAMES = [
    "unit_number",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def fetch_and_extract_cmapss(
    dest_dir: str | Path = "data/raw",
    sample_output_path: str | Path = "data/samples/fd001_train_5engines.csv",
) -> Path:
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    zip_path = dest_path / "CMAPSSData.zip"

    if not zip_path.exists():
        print(f"Downloading C-MAPSS dataset from {CMAPSS_URL}...")
        req = urllib.request.Request(CMAPSS_URL, headers={"User-Agent": "Aegis-Dataset-Fetcher/1.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        zip_path.write_bytes(data)
    else:
        print(f"Using cached zip at {zip_path}")
        data = zip_path.read_bytes()

    # Checksum verification
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"Checksum verification failed for {zip_path}!\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_hash}"
        )
    print("Checksum verified successfully.")

    # Extract train_FD001.txt
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if "train_FD001.txt" not in z.namelist():
            raise FileNotFoundError("train_FD001.txt not found in CMAPSSData.zip archive.")
        raw_txt = z.read("train_FD001.txt").decode("utf-8")

    # Parse whitespace-separated table
    df = pd.read_csv(io.StringIO(raw_txt), sep=r"\s+", header=None)
    # The raw file may have trailing whitespace creating extra columns
    df = df.iloc[:, : len(COLUMN_NAMES)]
    df.columns = COLUMN_NAMES

    # Filter engines 1 to 5
    df_5engines = df[df["unit_number"] <= 5].copy()

    # Save to data/samples/
    out_file = Path(sample_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_5engines.to_csv(out_file, index=False)
    print(f"Wrote {len(df_5engines)} rows for engines 1-5 to {out_file}")
    return out_file


def main():
    fetch_and_extract_cmapss()


if __name__ == "__main__":
    main()

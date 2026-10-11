#!/usr/bin/env python3
"""Verify the paper data within the original-style verl/ and math_evaluation/ tree."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TRAIN_COLUMNS = {
    "sample_id", "problem_id", "question", "ground_truth", "response",
    "prediction", "is_correct", "data_source", "original_source", "extra_info",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT, help="New repository root")
    parser.add_argument("--quick", action="store_true", help="Skip SHA-256 checks")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    manifest = json.loads((data_root / "verl" / "train_configs" / "data_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema version")
    checked = 0
    for group in ("train", "eval", "prompts"):
        for name, record in manifest[group].items():
            path = (data_root / record["path"]).resolve()
            if data_root not in path.parents or not path.is_file():
                raise FileNotFoundError(f"Missing/unsafe path: {path}")
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"Size mismatch: {path}")
            if not args.quick and sha256(path) != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}")
            if group == "train":
                parquet = pq.ParquetFile(path)
                if parquet.metadata.num_rows != record["rows"]:
                    raise ValueError(f"Row count mismatch: {path}")
                if set(parquet.schema_arrow.names) != TRAIN_COLUMNS:
                    raise ValueError(f"Train schema mismatch: {path}")
                field = parquet.schema_arrow.field("extra_info").type
                if {item.name for item in field} != {"question", "answer"}:
                    raise ValueError(f"Missing verl SFT prompt/response keys: {path}")
            else:
                with path.open(encoding="utf-8") as stream:
                    count = sum(1 for line in stream if json.loads(line))
                if count != record["rows"]:
                    raise ValueError(f"Row count mismatch: {path}")
            checked += 1
            print(f"OK {group}/{name}: {record['rows']} rows")
    if manifest["train"]["correctness_positive"]["rows"] != 22222:
        raise ValueError("Unexpected Figure 1 positive sample count")
    if manifest["train"]["correctness_negative"]["rows"] != 25778:
        raise ValueError("Unexpected Figure 1 negative sample count")
    if manifest["train"]["easy_444"]["unique_problems"] != 444:
        raise ValueError("Expected 444 solvable prompts")
    if manifest["eval"]["math"]["rows"] != 2000:
        raise ValueError("Expected 2000 MATH evaluation examples")
    print(f"Verified {checked} dataset files and metadata. quick={args.quick}")


if __name__ == "__main__":
    main()

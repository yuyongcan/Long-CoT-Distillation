#!/usr/bin/env python3
"""Copy selected paper datasets from the adjacent DFT checkout into the original-style layout.

All output paths are under the new repository. The source DFT checkout remains unchanged.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT.parent / "DFT"
MANIFEST_PATH = REPO_ROOT / "verl" / "train_configs" / "data_manifest.json"
TRAIN_OUTPUTS = {
    "correctness_positive": "verl/data/dapo17k_qwen3_4b_4k/train_pos.parquet",
    "correctness_negative": "verl/data/dapo17k_qwen3_4b_4k/train_neg.parquet",
    "diversity_500": "verl/data/prompt_diversity/train_pos_500.parquet",
    "diversity_16k": "verl/data/prompt_diversity/train_pos_16k.parquet",
    "easy_444": "verl/data/easy_444/train_pos.parquet",
    "complexity_simple": "verl/data/numina_cot/train.parquet",
    "complexity_complex": "verl/data/dapo17k_qwen3_4b_32k/train_pos.parquet",
}
PROMPT_OUTPUTS = {
    "random_3k": "math_evaluation/data/dapo3k_train/test.jsonl",
    "diversity_500": "math_evaluation/data/dapo500/test.jsonl",
    "diversity_16k": "math_evaluation/data/dapo16k/test.jsonl",
    "easy_444": "math_evaluation/data/dapo_easy_444/test.jsonl",
}
INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
SCHEMA = pa.schema(
    [
        ("sample_id", pa.int64()),
        ("problem_id", pa.int64()),
        ("question", pa.string()),
        ("ground_truth", pa.string()),
        ("response", pa.string()),
        ("prediction", pa.string()),
        ("is_correct", pa.bool_()),
        ("data_source", pa.string()),
        ("original_source", pa.string()),
        ("extra_info", pa.struct([("question", pa.string()), ("answer", pa.string())])),
    ]
)
# The source names are recorded verbatim in the manifest. Some historical directory
# names say "no_think_32768", although the actual responses contain <think> tags.
TRAIN_SOURCES = {
    "correctness_positive": ("verl/data/dapo17k_Qwen3-4B_qwen3_no_think_4096/train_pos.parquet", 22222, True),
    "correctness_negative": ("verl/data/dapo17k_Qwen3-4B_qwen3_no_think_4096/train_neg.parquet", 25778, False),
    "diversity_500": ("verl/data/dapo_scale_pos/train_pos_500.parquet", 29447, True),
    "diversity_16k": ("verl/data/dapo_scale_pos/train_pos_16k.parquet", 29494, True),
    "easy_444": ("verl/data/dapo17k_Qwen3-4B_plusk7_Qwen3-8B-Base/train_pos.parquet", 26445, True),
    "complexity_simple": ("verl/data/numina_cot_with_suffix/train.parquet", 100000, None),
    "complexity_complex": ("verl/data/dapo17k_Qwen3-4B_qwen3_no_think_32768/train_pos.parquet", 20189, True),
}
EVAL_SOURCES = {
    "aime24": "math_evaluation/data/aime24/test.jsonl",
    "amc23": "math_evaluation/data/amc23/test.jsonl",
    "minerva_math": "math_evaluation/data/minerva_math/test.jsonl",
    "olympiadbench": "math_evaluation/data/olympiadbench/test.jsonl",
    "math": "math_evaluation/data/math/test.jsonl",
}
PROMPT_SOURCES = {
    "random_3k": "math_evaluation/data/dapo3k_train/test.jsonl",
    "diversity_500": "math_evaluation/data/dapo500/test.jsonl",
    "diversity_16k": "math_evaluation/data/dapo16k/test.jsonl",
    "easy_444": "math_evaluation/data/dapo_plusk7_Qwen3-8B-Base/test.jsonl",
}
# Restrict this scan to concrete internal path/secret patterns; ordinary mathematics
# text can contain URLs and must not be altered silently.
PRIVATE_PATTERNS = (
    re.compile(r"/mnt/dolphinfs/|/home/hadoop-|/root/\.ssh/"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
)


def check_public_text(value: str, location: str) -> None:
    for pattern in PRIVATE_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"Potential internal path or credential in {location}; review the source before release")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize(row: dict, sample_id: int, expected_label: bool | None) -> dict:
    extra = row.get("extra_info") or {}
    question = row.get("question")
    if question is None:
        # NuminaMath: extra_info.question already contains the original prompt suffix.
        question = extra["question"]
    question = str(question)
    prompt = extra.get("question", question + " " + INSTRUCTION)
    response = row.get("response", extra.get("answer"))
    if not isinstance(response, str) or not isinstance(prompt, str) or not question:
        raise ValueError(f"Invalid question/response in source row {sample_id}")
    gt = row.get("gt")
    if gt is None:
        gt = (row.get("reward_model") or {}).get("ground_truth")
    problem_id = row.get("idx", extra.get("original_idx", extra.get("index")))
    if problem_id is None:
        raise ValueError(f"Missing original problem ID in source row {sample_id}")
    label = row.get("label", extra.get("label"))
    if expected_label is not None and label is not None:
        if str(label).lower() not in ({"true", "correct", "1"} if expected_label else {"false", "incorrect", "0"}):
            raise ValueError(f"Unexpected label for source row {sample_id}: {label!r}")
    pred = row.get("pred")
    source = row.get("source")  # NuminaMath's public sub-source (not its local directory).
    normalized = {
        "sample_id": sample_id,
        "problem_id": int(problem_id),
        "question": question,
        "ground_truth": str(gt) if gt is not None else None,
        "response": response,
        "prediction": str(pred) if pred is not None else None,
        "is_correct": expected_label,
        "data_source": "AI-MO/NuminaMath-CoT" if expected_label is None else "DAPO-Math-17K/Qwen3-4B",
        "original_source": str(source) if source is not None else None,
        # Keep the exact prompt and answer previously fed to verl, including suffixes.
        "extra_info": {"question": prompt, "answer": response},
    }
    for key in ("question", "ground_truth", "response", "prediction", "original_source"):
        check_public_text(normalized[key] or "", f"row {sample_id}, field {key}")
    check_public_text(prompt, f"row {sample_id}, training prompt")
    return normalized


def prepare_train(source: Path, output: Path, expected_rows: int, expected_label: bool | None) -> dict:
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"{source}: expected {expected_rows} rows, got {parquet.metadata.num_rows}")
    output.parent.mkdir(parents=True, exist_ok=True)
    unique_ids: set[int] = set()
    null_gt = 0
    sample_id = 0
    with pq.ParquetWriter(output, SCHEMA, compression="zstd") as writer:
        for batch in parquet.iter_batches(batch_size=512):
            cleaned = []
            for row in batch.to_pylist():
                entry = normalize(row, sample_id, expected_label)
                unique_ids.add(entry["problem_id"])
                null_gt += entry["ground_truth"] is None
                cleaned.append(entry)
                sample_id += 1
            writer.write_table(pa.Table.from_pylist(cleaned, schema=SCHEMA))
    return {"rows": sample_id, "unique_problems": len(unique_ids), "null_ground_truth": null_gt}


def prepare_eval(source: Path, output: Path, name: str) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_indices: list[int] | None = None
    if name == "math":
        with source.open(encoding="utf-8") as stream:
            total = sum(1 for line in stream if line.strip())
        if total != 5000:
            raise ValueError(f"Expected 5000 MATH test problems, found {total}")
        selected_indices = sorted(random.Random(42).sample(range(total), 2000))
    else:
        total = None
    count = 0
    selected = set(selected_indices or ())
    with source.open(encoding="utf-8") as stream, output.open("w", encoding="utf-8") as target:
        for original_idx, line in enumerate(stream):
            if not line.strip():
                raise ValueError(f"Empty JSONL line in {source}: {original_idx + 1}")
            if selected_indices is not None and original_idx not in selected:
                continue
            entry = json.loads(line)
            for key, value in entry.items():
                if isinstance(value, str):
                    check_public_text(value, f"{name} test item {original_idx}, {key}")
            if selected_indices is not None:
                entry["idx"] = original_idx
            target.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    if selected_indices is not None and count != 2000:
        raise ValueError(f"MATH subset has {count} rows, expected 2000")
    result = {"rows": count}
    if total is not None:
        result.update({"original_rows": total, "sample_seed": 42, "sample_indices": selected_indices})
    return result


def ensure_new_target(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"{output} already exists; use an empty output directory to avoid overwriting data")


def already_staged(output: Path, section: dict, name: str, resume: bool) -> bool:
    if not resume or not output.exists():
        ensure_new_target(output)
        if resume and name in section:
            raise FileNotFoundError(f"Previously staged dataset missing: {output}")
        return False
    previous = section.get(name)
    if previous is None or previous.get("sha256") != digest(output):
        raise ValueError(f"Cannot resume: {output} is untracked or its SHA-256 has changed")
    print(f"Already staged {name}: checksum verified", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT, help="Existing local DFT repository")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT, help="New public repository root")
    parser.add_argument("--resume", action="store_true", help="Verify the existing copied data without overwriting")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = output_root / "verl" / "train_configs" / "data_manifest.json"
    if args.resume and not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot resume without existing manifest: {manifest_path}")
    if not args.resume:
        ensure_new_target(manifest_path)
    for rel_path, _, _ in TRAIN_SOURCES.values():
        if not (source_root / rel_path).is_file():
            raise FileNotFoundError(source_root / rel_path)
    for rel_path in (*EVAL_SOURCES.values(), *PROMPT_SOURCES.values()):
        if not (source_root / rel_path).is_file():
            raise FileNotFoundError(source_root / rel_path)

    manifest: dict = json.loads(manifest_path.read_text(encoding="utf-8")) if args.resume else {
        "schema_version": 1,
        "selection": "Representative experiments from Figures 1, 3, 4, and 5",
        "train": {},
        "eval": {},
        "prompts": {},
        "notes": [
            "Full training trajectories are copied for the selected conditions; model weights are not included.",
            "The 4k positive trajectories are reused for Random-3k and Medium.",
            "The fixed MATH-2000 subset is a new seed-42 draw, not the paper's preserved IDs.",
        ],
    }
    if manifest["schema_version"] != 1:
        raise ValueError("Incompatible data manifest schema")
    manifest.setdefault("prompts", {})
    for name, (rel_path, expected_rows, expected_label) in TRAIN_SOURCES.items():
        source, output = source_root / rel_path, output_root / TRAIN_OUTPUTS[name]
        if already_staged(output, manifest["train"], name, args.resume):
            continue
        print(f"Curating {name} ({expected_rows} rows) ...", flush=True)
        stats = prepare_train(source, output, expected_rows, expected_label)
        manifest["train"][name] = {
            "path": str(output.relative_to(output_root)),
            "source": rel_path,
            "source_sha256": digest(source),
            "sha256": digest(output),
            "bytes": output.stat().st_size,
            **stats,
        }

    for name, rel_path in EVAL_SOURCES.items():
        source, output = source_root / rel_path, output_root / "math_evaluation" / "data" / name / "test.jsonl"
        if already_staged(output, manifest["eval"], name, args.resume):
            continue
        print(f"Curating evaluation set {name} ...", flush=True)
        stats = prepare_eval(source, output, name)
        manifest["eval"][name] = {
            "path": str(output.relative_to(output_root)),
            "source": rel_path,
            "source_sha256": digest(source),
            "sha256": digest(output),
            "bytes": output.stat().st_size,
            **stats,
        }
    for name, rel_path in PROMPT_SOURCES.items():
        source, output = source_root / rel_path, output_root / PROMPT_OUTPUTS[name]
        if already_staged(output, manifest["prompts"], name, args.resume):
            continue
        print(f"Curating generation prompts {name} ...", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with source.open(encoding="utf-8") as stream, output.open("w", encoding="utf-8") as target:
            for line in stream:
                row = json.loads(line)
                for value in row.values():
                    if isinstance(value, str):
                        check_public_text(value, f"prompts/{name} row {count}")
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        manifest["prompts"][name] = {
            "path": str(output.relative_to(output_root)),
            "source": rel_path,
            "source_sha256": digest(source),
            "sha256": digest(output),
            "bytes": output.stat().st_size,
            "rows": count,
        }
    # A stable, repo-relative manifest works as an upload inventory and as an
    # independent checksum list after downloading the Hugging Face Dataset.
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Staged {len(TRAIN_SOURCES)} full train conditions and {len(EVAL_SOURCES)} benchmarks in {output_root}")


if __name__ == "__main__":
    main()

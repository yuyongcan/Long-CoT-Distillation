#!/usr/bin/env python3
"""Run the five-paper-benchmark math evaluation at avg@16.

All paths come from the selected local dataset folder; no downloads are attempted.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "verl" / "train_configs" / "experiments.yaml"
ENGINE = ROOT / "math_evaluation" / "math_eval_v2.py"


def latest_model_checkpoint(experiment: str, checkpoint_root: Path) -> Path:
    path = checkpoint_root / experiment
    checkpoints = [p for p in path.glob("global_step_*") if p.is_dir() and p.name[12:].isdigit()]
    if not checkpoints:
        raise FileNotFoundError(f"No training checkpoint found under {path}. Pass --model-path explicitly.")
    return max(checkpoints, key=lambda p: int(p.name[12:]))


def eval_command(config: dict, model_path: Path, data_root: Path, output_dir: Path, max_tokens: int, tensor_parallel_size: int) -> list[str]:
    settings = config["evaluation"]
    return [
        sys.executable, str(ENGINE),
        "--model_name_or_path", str(model_path),
        "--data_dir", str(data_root / "math_evaluation" / "data"),
        "--output_dir", str(output_dir),
        "--data_names", ",".join(settings["benchmarks"]),
        "--split", "test",
        "--prompt_type", settings["prompt_type"],
        "--n_sampling", str(settings["n_sampling"]),
        "--temperature", str(settings["temperature"]),
        "--top_p", str(settings["top_p"]),
        "--max_tokens_per_call", str(max_tokens),
        "--seed", str(settings["seed"]),
        "--tensor_parallel_size", str(tensor_parallel_size),
        "--start", "0", "--end", "-1",
        "--use_vllm",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", help="Experiment name from verl/train_configs/experiments.yaml")
    parser.add_argument("--model-path", type=Path, help="Trained model dir (overrides the experiment checkpoint)")
    parser.add_argument("--data-root", type=Path, default=ROOT, help="Repository root containing math_evaluation/data/")
    parser.add_argument("--output-root", type=Path, default=ROOT / "math_evaluation" / "outputs")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "verl" / "checkpoints")
    parser.add_argument("--max-new-tokens", type=int, help="Override 4096 or 32768 for the complex dataset")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.experiment and not args.model_path:
        parser.error("Pass --experiment or --model-path")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    selected = config["experiments"].get(args.experiment)
    if args.experiment and selected is None:
        parser.error(f"Unknown experiment: {args.experiment}")
    data_root = args.data_root.resolve()
    manifest_path = data_root / "verl" / "train_configs" / "data_manifest.json"
    if not manifest_path.is_file():
        parser.error(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        parser.error("Unsupported data manifest schema version")
    for name in config["evaluation"]["benchmarks"]:
        record = manifest["eval"][name]
        test_file = (data_root / record["path"]).resolve()
        if not test_file.is_file() or data_root not in test_file.parents or test_file.stat().st_size != record["bytes"]:
            parser.error(f"Missing or changed benchmark: {test_file}")
        if name == "math" and record["rows"] != config["evaluation"]["math_subset_size"]:
            parser.error("MATH subset count does not match the evaluation config")
    model_path = args.model_path.resolve() if args.model_path else latest_model_checkpoint(args.experiment, args.checkpoint_root)
    if not model_path.is_dir():
        parser.error(f"Model directory does not exist: {model_path}")
    run_name = args.experiment or model_path.name
    output_dir = (args.output_root / run_name).resolve()
    if output_dir.exists():
        parser.error(f"Evaluation output exists; refusing to overwrite: {output_dir}")
    max_tokens = args.max_new_tokens or (
        config["evaluation"]["max_new_tokens_complex"] if selected and selected["dataset"] == "complexity_complex"
        else config["evaluation"]["max_new_tokens"]
    )
    if max_tokens < 1 or args.tensor_parallel_size < 1:
        parser.error("Token budget and tensor-parallel size must be positive")
    cmd = eval_command(config, model_path, data_root, output_dir, max_tokens, args.tensor_parallel_size)
    print("Command:", shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    output_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # The MIT-licensed parser is installed from the vendored latex2sympy directory.
    subprocess.run(cmd, cwd=ROOT / "math_evaluation", env=env, check=True)
    metrics = {}
    for name in config["evaluation"]["benchmarks"]:
        metrics[name] = json.loads((output_dir / f"{name}_metrics.json").read_text(encoding="utf-8"))
    results = {"metric": "avg@16", "benchmarks": {name: item["avg_at_k"] for name, item in metrics.items()}}
    results["mean_accuracy"] = round(sum(results["benchmarks"].values()) / len(metrics), 2)
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

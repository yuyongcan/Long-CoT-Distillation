#!/usr/bin/env python3
"""Train a selected paper configuration using the copied verl SFT trainer."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("experiments.yaml")


def load_config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Unsupported experiments.yaml version")
    return config


def experiment_data(config: dict, name: str, data_root: Path) -> tuple[dict, Path, dict]:
    experiments = config["experiments"]
    if name not in experiments:
        raise ValueError(f"Unknown experiment {name!r}. Choose one of: {', '.join(experiments)}")
    experiment = experiments[name]
    manifest_path = data_root / "verl" / "train_configs" / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing data manifest: {manifest_path}. Place data in verl/data first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported data manifest schema version")
    data_entry = manifest["train"][experiment["dataset"]]
    data_file = (data_root / data_entry["path"]).resolve()
    if not data_file.is_file() or data_file.stat().st_size != data_entry["bytes"]:
        raise FileNotFoundError(f"Training data missing or size changed: {data_file}")
    if config["train"]["training_files_are_read_only"] and data_root.resolve() not in data_file.parents:
        raise ValueError("Training data path escapes the dataset directory")
    if config["train"]["ulysses_sequence_parallel_size"] != 1:
        raise ValueError("Both losses require sequence_parallel_size=1 in this release")
    return experiment, data_file, data_entry


def make_command(config: dict, name: str, model: str, data_file: Path, output: Path, gpus: int) -> list[str]:
    experiment = config["experiments"][name]
    train = config["train"]
    if gpus < 1:
        raise ValueError("gpus must be positive")
    if train["global_batch_size"] % (gpus * experiment["micro_batch_per_gpu"]):
        raise ValueError("Global batch must be divisible by GPUs × micro_batch_per_gpu")
    module = "verl.trainer.fsdp_sft_trainer" if experiment["loss"] == "logp" else "verl.trainer.fsdp_dft_trainer"
    if experiment["loss"] not in {"logp", "p"}:
        raise ValueError("Unknown training loss: " + str(experiment["loss"]))
    return [
        "torchrun", "--standalone", "--nnodes=1", f"--nproc_per_node={gpus}", "-m", module,
        f"data.train_files={data_file}",
        "data.val_files=null",  # No benchmark is reused as training validation data.
        "data.prompt_key=extra_info", "data.response_key=extra_info",
        "data.prompt_dict_keys=[question]", "data.response_dict_keys=[answer]",
        f"data.train_batch_size={train['global_batch_size']}",
        f"data.micro_batch_size_per_gpu={experiment['micro_batch_per_gpu']}",
        f"data.max_length={experiment['max_length']}",
        f"data.truncation={train['truncation']}",
        f"optim.lr={train['learning_rate']}",
        f"optim.betas=[{','.join(str(x) for x in train['optimizer_betas'])}]",
        f"optim.weight_decay={train['weight_decay']}",
        f"optim.lr_scheduler={train['lr_scheduler']}",
        f"optim.warmup_steps_ratio={train['warmup_steps_ratio']}",
        f"optim.clip_grad={train['gradient_clip']}",
        f"model.partial_pretrain={model}",
        f"model.strategy={train['fsdp_strategy']}",
        f"model.fsdp_config.model_dtype={train['dtype']}",
        f"model.fsdp_config.cpu_offload={str(experiment.get('cpu_offload', False)).lower()}",
        f"model.fsdp_config.offload_params={str(experiment.get('cpu_offload', False)).lower()}",
        f"model.use_liger={str(train['use_liger']).lower()}", "model.enable_gradient_checkpointing=true",
        f"trainer.default_local_dir={output}",
        f"trainer.project_name={config['project']}",
        f"trainer.experiment_name={name}",
        f"trainer.logger=[console,tensorboard]",
        "trainer.default_hdfs_dir=null", "trainer.test_freq=-1",
        f"trainer.save_freq={train['save_frequency']}",
        f"trainer.total_epochs={experiment.get('epochs', train['epochs'])}",
        f"trainer.seed={train['seed']}",
        f"ulysses_sequence_parallel_size={train['ulysses_sequence_parallel_size']}",
        f"use_remove_padding={str(train['use_remove_padding']).lower()}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List experiment names")
    parser.add_argument("--experiment", help="Name from verl/train_configs/experiments.yaml")
    parser.add_argument("--model-path", help="Local HF checkpoint; defaults to configured Hub ID")
    parser.add_argument("--data-root", type=Path, default=ROOT, help="Repository root with verl/data/")
    parser.add_argument("--output-root", type=Path, default=ROOT / "verl" / "checkpoints")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args = parser.parse_args()
    config = load_config()
    if args.list:
        for name in config["experiments"]:
            print(name)
        return
    if not args.experiment:
        parser.error("--experiment is required unless --list is used")
    data_root = args.data_root.resolve()
    experiment, data_file, info = experiment_data(config, args.experiment, data_root)
    model = args.model_path or config["models"][experiment["model"]]
    if args.model_path and not Path(args.model_path).is_dir():
        parser.error("--model-path must point to an existing HF model directory")
    output = (args.output_root / args.experiment).resolve()
    if output.exists():
        parser.error(f"Output already exists; refusing to overwrite or resume: {output}")
    gpus = args.gpus if args.gpus is not None else config["train"]["gpus"]
    cmd = make_command(config, args.experiment, model, data_file, output, gpus)
    print(f"Experiment: {args.experiment} ({experiment['paper']})", flush=True)
    print(f"Train: {data_file} ({info['rows']} trajectories, {info['unique_problems']} problems)", flush=True)
    print("Command:", shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if not args.model_path:
        parser.error("Pass --model-path to avoid silently attempting a network download")
    if not (ROOT / "verl" / "verl" / "__init__.py").is_file():
        parser.error("Missing verl package")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "verl") + os.pathsep + env.get("PYTHONPATH", "")
    env["TENSORBOARD_DIR"] = str(ROOT / "verl" / "tensorboard_log" / args.experiment)
    env["HF_HUB_OFFLINE"] = "1" if args.model_path else env.get("HF_HUB_OFFLINE", "0")
    subprocess.run(cmd, cwd=ROOT / "verl", env=env, check=True)


if __name__ == "__main__":
    main()

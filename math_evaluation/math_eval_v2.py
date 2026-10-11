"""
Modified for Long-CoT-Distillation: portable paths and deterministic --shuffle.
math_eval_v2.py  —— 多卡并行评估（subprocess 方案）

调度模式（默认，不传 --rank）：
    主进程检测 GPU 数量，为每张卡 fork 一个独立 Python 子进程，
    子进程通过 CUDA_VISIBLE_DEVICES=<单卡号> 启动，完全隔离 CUDA 上下文，
    不存在 "Cannot re-initialize CUDA in forked subprocess" 或 OOM 问题。

Worker 模式（传 --rank N）：
    直接在当前进程做推理，CUDA_VISIBLE_DEVICES 已由调度器设好。
"""
import random
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import json

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    repository_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data_dir", default=str(repository_root / "math_evaluation" / "data"), type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--output_dir", default=str(repository_root / "math_evaluation" / "outputs"), type=str)
    parser.add_argument("--tensor_parallel_size", default=1, type=int)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=8192, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--adapt_few_shot", action="store_true")
    parser.add_argument("--skip_scoring", action="store_true", help="Generation only; save raw rollouts without scoring")
    # worker 专用参数：调度器传入，标识本进程是哪张卡的 worker
    parser.add_argument("--rank", type=int, default=-1,
                        help="Worker rank (GPU index). -1 means dispatcher mode.")
    parser.add_argument("--world_size", type=int, default=-1)
    parser.add_argument("--data_name_single", type=str, default="",
                        help="Single dataset name for worker mode.")
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    if args.shuffle:
        random.Random(args.seed).shuffle(examples)
        print('Shuffle Successfully')

    examples = examples[args.start : len(examples) if args.end == -1 else args.end]
    return examples


# ──────────────────────────────────────────────
# Worker 模式：单张 GPU 上做推理
# ──────────────────────────────────────────────
def run_worker(args):
    """在当前进程（已设好 CUDA_VISIBLE_DEVICES）里完成推理并保存结果。"""
    # 延迟导入，让 CUDA_VISIBLE_DEVICES 在 import 前生效
    import torch
    from vllm import LLM
    from transformers import AutoTokenizer
    from model_utils import load_hf_lm_and_tokenizer

    rank = args.rank
    world_size = args.world_size
    data_name = args.data_name_single

    print(f"[Worker {rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    # 准备数据（worker 模式下 start/end 已由调度器写入 args）
    examples = prepare_data(data_name, args)
    # 按 rank 切片
    chunk_size = len(examples) // world_size
    start_idx = rank * chunk_size
    end_idx = len(examples) if rank == world_size - 1 else (rank + 1) * chunk_size
    examples_chunk = examples[start_idx:end_idx]
    print(f"[Worker {rank}] 处理 {len(examples_chunk)} 条样本 (idx {start_idx}~{end_idx-1})")

    torch.manual_seed(args.seed + rank)

    # gemma 系列权重在 fp16 下会数值溢出导致 logits NaN，
    # 采样退化为持续输出 <pad>(token 0)。这里强制使用 bfloat16。
    vllm_dtype = "bfloat16" if "gemma" in args.model_name_or_path.lower() else "auto"
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=1,
            trust_remote_code=True,
            max_model_len=min(32768, args.max_tokens_per_call + 1000),
            dtype=vllm_dtype,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    result = _infer(llm, tokenizer, data_name, args, examples_chunk, rank)

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{data_name}_results_rank_{rank}.json")
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Worker {rank}] 完成，结果写入 {out_file}")


# ──────────────────────────────────────────────
# 调度器模式：为每张 GPU 启动一个独立子进程
# ──────────────────────────────────────────────
def run_dispatcher(args):
    import torch

    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个GPU")
    if world_size == 0:
        raise RuntimeError("没有检测到可用的GPU")
    if args.tensor_parallel_size < 1 or world_size % args.tensor_parallel_size:
        raise ValueError("GPU count must be divisible by --tensor_parallel_size")
    if args.pipeline_parallel_size != 1:
        raise ValueError("Pipeline parallelism is not supported by the subprocess evaluator")
    num_workers = world_size // args.tensor_parallel_size

    # 解析父进程的 CUDA_VISIBLE_DEVICES
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None:
        gpu_ids = [g.strip() for g in visible.split(",")]
    else:
        gpu_ids = [str(i) for i in range(world_size)]
    if len(gpu_ids) != world_size:
        raise ValueError("CUDA_VISIBLE_DEVICES count does not match torch.cuda.device_count()")

    data_list = args.data_names.split(",")
    all_results = []

    for data_name in data_list:
        print("=" * 50)
        print(f"数据集: {data_name}")

        # 收集当前命令行原始参数（去掉已有的 --rank/--world_size/--data_name_single）
        base_argv = [a for a in sys.argv[1:]
                     if not a.startswith("--rank")
                     and not a.startswith("--world_size")
                     and not a.startswith("--data_name_single")]

        procs = []
        for rank in range(num_workers):
            worker_gpus = gpu_ids[rank * args.tensor_parallel_size: (rank + 1) * args.tensor_parallel_size]
            cmd = [
                sys.executable, __file__,
                *base_argv,
                "--rank", str(rank),
                "--world_size", str(num_workers),
                "--data_name_single", data_name,
            ]
            env = os.environ.copy()
            # ★ 关键：在子进程环境中只暴露单张卡，子进程 import torch/vllm 时只看到这张卡
            env["CUDA_VISIBLE_DEVICES"] = ",".join(worker_gpus)
            print(f"启动 Worker {rank}，GPU={worker_gpus}，命令: {' '.join(cmd[:6])}...")
            p = subprocess.Popen(cmd, env=env)
            procs.append(p)

        # 等待所有 worker 完成
        for rank, p in enumerate(procs):
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"Worker {rank} 异常退出，returncode={ret}")

        # 合并结果
        result_json = merge_results(args, data_name, num_workers)
        all_results.append(result_json)

    # 汇总 avg
    if args.skip_scoring:
        return
    data_list.append("avg")
    avg_result = {"acc": sum(r["acc"] for r in all_results) / len(all_results)}
    if all("pass_at_k" in r for r in all_results):
        max_k = max(len(r["pass_at_k"]) for r in all_results)
        avg_pass_at_k = []
        for k_idx in range(max_k):
            vals = [r["pass_at_k"][k_idx] for r in all_results if k_idx < len(r["pass_at_k"])]
            avg_pass_at_k.append(round(sum(vals) / len(vals), 1))
        avg_result["pass_at_k"] = avg_pass_at_k
    all_results.append(avg_result)
    print("=" * 50)
    print("所有数据集评估完成")
    for name, res in zip(data_list, all_results):
        print(f"  {name}: {res}")


def merge_results(args, data_name, world_size):
    print(f"合并 {data_name} 的结果...")
    all_samples = []
    for rank in range(world_size):
        result_file = os.path.join(args.output_dir, f"{data_name}_results_rank_{rank}.json")
        with open(result_file, "r", encoding="utf8") as f:
            result = json.load(f)
            all_samples.extend(result["samples"])
        os.remove(result_file)

    if args.skip_scoring:
        result_json = {"num_samples": len(all_samples), "scored": False}
    else:
        all_samples, result_json = evaluate(
            samples=all_samples,
            data_name=data_name,
            prompt_type=args.prompt_type,
            execute=True,
        )
    with open(os.path.join(args.output_dir, f"{data_name}_final_results.json"), "w", encoding="utf8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, f"{data_name}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)

    print(f"{data_name} 合并完成，共 {len(all_samples)} 条样本，结果: {result_json}")
    return result_json


# ──────────────────────────────────────────────
# 推理核心（worker 内部调用）
# ──────────────────────────────────────────────
def is_multi_choice(answer):
    if not isinstance(answer, str):
        return False
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def _infer(llm, tokenizer, data_name, args, examples, rank):
    from vllm import SamplingParams
    from model_utils import generate_completions

    executor = PythonExecutor(get_answer_expr="solution()") if args.prompt_type in ("pal", "pot") else None

    samples = []
    for example in tqdm(examples, total=len(examples), desc=f"GPU {rank}"):
        idx = example["idx"]
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)
        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }
        for key in ["level", "type", "unit", "solution_type", "choices", "solution",
                    "ques_type", "ans_type", "answer_type", "dataset", "subfield",
                    "filed", "theorem", "answer"]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    input_prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]

    print('start apply chat template')
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    print('end apply chat template')

    if not input_prompts:
        return {"samples": [], "time_use_in_second": 0.0, "time_use_in_minute": "0:00"}

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>",
                  "<|end_of_text|>", "<｜end▁of▁sentence｜>"]
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    if "qwen2" in args.model_name_or_path.lower():
        stop_token_ids = [151645, 151643]
    elif "deepseek" in args.model_name_or_path.lower():
        stop_token_ids = [100001]
    else:
        stop_token_ids = None

    start_time = time.time()

    if args.use_vllm:
        outputs = llm.generate(
            input_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_words,
                stop_token_ids=stop_token_ids,
            ),
        )
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        output_token_lens = [len(o.outputs[0].token_ids) for o in outputs]
        outputs = [o.outputs[0].text for o in outputs]
    else:
        outputs = generate_completions(
            model=llm,
            tokenizer=tokenizer,
            prompts=input_prompts,
            max_new_tokens=args.max_tokens_per_call,
            batch_size=16,
            stop_id_sequences=stop_words,
        )
        output_token_lens = [len(tokenizer.encode(o)) for o in outputs]

    assert len(outputs) == len(input_prompts)

    codes = []
    for i in range(len(input_prompts)):
        code = outputs[i].rstrip()
        for sw in stop_words:
            if sw in code:
                code = code.split(sw)[0].strip()
        codes.append(code)

    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]
    time_use = time.time() - start_time

    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling: (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling: (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in ["A", "B", "C", "D", "E"]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                preds[j] = "".join(c for c in (preds[j] or "") if c in ["A", "B", "C", "D", "E"])
        token_lens = output_token_lens[i * args.n_sampling: (i + 1) * args.n_sampling]
        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports, "token_len": token_lens})
        all_samples.append(sample)

    return {
        "samples": all_samples,
        "time_use_in_second": time_use,
        "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}",
    }


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if args.rank == -1:
        # ── 调度器模式 ──
        print("=" * 50)
        print("Running with arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print("=" * 50)
        run_dispatcher(args)
    else:
        # ── Worker 模式（由调度器通过 subprocess 启动，CUDA_VISIBLE_DEVICES 已设好）──
        run_worker(args)

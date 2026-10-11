# Modified for Long-CoT-Distillation: configurable worker count, strict avg@k.
import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import os

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0

    worker_count = min(int(os.environ.get("MATH_EVAL_WORKERS", "8")), os.cpu_count() or 1)
    with ProcessPool(max_workers=worker_count) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(params), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    raise RuntimeError(f"Math scoring failed for {data_name}") from error
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])
    if any(len(s) != max_len for s in score_mat):
        raise ValueError("Every problem must have the same number of samples for avg@k")

    # output mean of each column of scores (avg@k)
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = [float(v) for v in np.round(col_means * 100, decimals=1)]

    mean_mean_score = round(float(np.mean(score_mat)) * 100, 1)
    print(f"mean_score (avg@k): {mean_score}")

    # compute pass@k: for each sample, check if at least one of the first k preds is correct
    pass_at_k = []
    for k in range(1, max_len + 1):
        pass_count = 0
        for s in score_mat:
            if any(s[:k]):
                pass_count += 1
        pass_at_k.append(round(pass_count / len(score_mat) * 100, 1))
    print(f"pass@k: {pass_at_k}")

    # compute average output token length
    all_token_lens = []
    for sample in samples:
        if 'token_len' in sample and sample['token_len']:
            all_token_lens.extend(sample['token_len'])
    avg_token_len = float(np.mean(all_token_lens)) if all_token_lens else None

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0],
        "all_acc": mean_score,
        "mean_acc": mean_mean_score,
        "avg_at_k": mean_mean_score,
        "pass_at_k": pass_at_k,
        "avg_token_len": round(avg_token_len, 1) if avg_token_len is not None else None,
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: float(np.round(np.array(v).mean() * 100, decimals=1)) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)

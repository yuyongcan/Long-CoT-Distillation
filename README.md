# Long-CoT Distillation

Code and representative experiments for **“Facts Don’t Speak Louder than Words: The Behavioral Essence of Long-CoT Distillation.”** This repository retains the original `DFT` layout (`verl/` for training and `math_evaluation/` for evaluation), with a smaller set of paper experiments and standardized paths. It excludes the original 2.7 TB of checkpoints. The original `DFT` directory was not modified.

## Paper overview

Long chain-of-thought (Long-CoT) distillation is often built around correct reasoning traces from a diverse set of problems. This paper asks whether correctness and broad prompt coverage are what actually drive the resulting gains. We propose the **Behavioral Hypothesis**: distillation primarily helps a model organize its existing knowledge into reusable, structured reasoning behaviors, rather than simply memorizing problem-specific facts.

### Incorrect answers can still teach reasoning behavior (Figure 1)

Fine-tuning on trajectories with **incorrect final answers** achieves **97.7% of the accuracy** of training on correct trajectories, retaining **94.7% of the improvement over the base models** when aggregated across the paper's mathematical comparisons (Section 3.1). Incorrect trajectories can also help when correct data is scarce: with 512 correct samples, increasing the positive-to-negative ratio from 1:0 to 1:5 raises mean accuracy across five math benchmarks from **20.4% to 26.4%** (Figure 2). These are not claims that arbitrarily corrupted reasoning is useful: the paper's incorrect set consists of teacher-generated trajectories selected by **final-answer correctness**, some of which retain useful reasoning structure.

![Figure 1 from the paper: accuracy after distillation on correct and incorrect trajectories across models, losses and math benchmarks.](assets/figure1-correctness.png)

### Prompt coverage is not the main bottleneck here (Figure 3)

Under a fixed candidate-trajectory budget, sampling from **500 prompts (128 responses each)** gives performance close to **16,000 prompts (4 responses each)**. For example, Qwen3-1.7B-Base reaches **28.0% vs. 26.8%** mean accuracy across five math benchmarks at the two endpoints (paper Table 9). These are candidate-generation settings; after correctness filtering, the corresponding released training files contain **450 vs. 10,729** distinct prompts.

![Figure 3 from the paper: math accuracy under different numbers of unique prompts and samples per prompt.](assets/figure3-prompt-diversity.png)

### Gains on problems the model already solves (Figure 4)

Fine-tuning on high-confidence, already-solvable problems (answered correctly in at least **7 of 8** base-model samples) still substantially improves performance and approaches training on randomly selected problems. The paper evaluates **Easy-444** for Qwen3-8B-Base and **Easy-643** for Qwen3-14B-Base; this repository includes the Easy-444 comparison. This is evidence **consistent with** behavioral alignment, not a parameter-level causal separation of behavior from knowledge.

![Figure 4 from the paper: performance after fine-tuning on already-solvable versus randomly selected problems.](assets/figure4-solvable-problems.png)

### The preferred SFT loss depends on CoT complexity (Figure 5)

For Qwen3-4B-Base, the paper reports the following `avg@16` accuracy averaged over five math benchmarks (Table 8). Mean token lengths describe the *training trajectories* (Section 5.1):

| CoT complexity | Mean training length | `-log p` | `-p` |
| --- | ---: | ---: | ---: |
| Simple | 426 tokens | 22.8% | **37.4%** |
| Medium | 1,337 tokens | 42.8% | **44.1%** |
| Complex | 9,062 tokens | **45.6%** | 40.5% |

The paper relates this contrast to token-gradient weighting: `-log p` gives relatively more weight to low-probability tokens than `-p`, potentially helping with difficult transitions in long reasoning chains. This explanation is an interpretation of the observed results, not a universal guarantee for either loss.

![Figure 5 from the paper: accuracy for probability-based and cross-entropy losses across Simple, Medium and Complex CoTs.](assets/figure5-loss-complexity.png)

These figures and numbers are from the **paper**, not measurements freshly reproduced by this repository. The released configurations cover representative comparisons only (e.g., the two Figure 3 endpoints, Easy-444 and Qwen3-4B-Base for Figure 5); the Figure 2 mixture study and some models and non-math evaluations are not included. The MATH-2000 subset bundled here is a new draw, so its evaluation results are not expected to exactly match the paper. See [Reproduce](#reproduce) for details.

## What's included

- `verl/verl/`: SFT trainers copied from `DFT/verl`, including `-log(p)` and `-p` (detached-probability-weighted NLL).
- `verl/train_configs/experiments.yaml` and `verl/train_configs/train.py`: selected experiments (Fig. 1: correct/incorrect; Fig. 3: 500 vs. 16k prompts; Fig. 4: 444 already-solvable vs. random problems; Fig. 5: simple/medium/complex CoTs with both losses).
- `verl/data/`: seven complete, representative training Parquet files in named experiment subdirectories.
- `math_evaluation/`: math evaluator copied from `DFT/math_evaluation`, with `data/` holding five benchmark test sets and four generation-prompt subsets; `sh/run_eval.py` launches evaluation.
- `verl/train_configs/data_manifest.json`: each copied data file's relative path, source, row count and SHA-256. Training and evaluation data directories are ignored by Git and can be distributed in a separate Hugging Face Dataset.

The **same positive 22,222-trajectory file** is reused for the random-problem and medium-CoT baselines; it is not duplicated. The negative set contains 25,778 trajectories. After filtering correct responses, the 500/16k-prompt datasets contain 450/10,729 distinct prompts with 29,447/29,494 trajectories. The easy set contains 444 prompts and 26,445 trajectories.

## Reproduce

Use Python 3.10 with PyTorch 2.6.0+cu124 (the existing local environments were checked for imports and data loading, **not** used to run full training). The paper used 8 A100-80G GPUs; the 32k-token experiments require substantial CPU memory. Install CUDA-compatible PyTorch, FlashAttention and Liger and the dependencies in `verl/requirements.txt`. Evaluation can use a separate Python 3.10 environment with `math_evaluation/requirements.txt` and `pip install --no-build-isolation -e math_evaluation/latex2sympy`. On this offline machine, use an existing prepared environment and local model checkpoints.

From the **new repository root**, run the following commands; replace model paths with existing local Hugging Face **Base** model directories. Each training command calls the same `torchrun -m verl.trainer.fsdp_sft_trainer` (`-log p`) or `fsdp_dft_trainer` (`-p`) pattern as the source scripts, using the matching file under `verl/data/`:

```bash
python verl/train_configs/verify_data.py
python verl/train_configs/train.py --list
python verl/train_configs/train.py --experiment fig1_correct_ce   --model-path /path/to/Qwen3-1.7B-Base --dry-run
python verl/train_configs/train.py --experiment fig1_correct_ce   --model-path /path/to/Qwen3-1.7B-Base
python verl/train_configs/train.py --experiment fig1_incorrect_ce --model-path /path/to/Qwen3-1.7B-Base
python verl/train_configs/train.py --experiment fig1_correct_p    --model-path /path/to/Qwen3-1.7B-Base
python verl/train_configs/train.py --experiment fig1_incorrect_p  --model-path /path/to/Qwen3-1.7B-Base
python math_evaluation/sh/run_eval.py --experiment fig1_correct_ce
```

For the other comparisons, select the named configuration and its matching Base model:

```bash
python verl/train_configs/train.py --experiment fig3_diversity_500 --model-path /path/to/Qwen3-1.7B-Base
python verl/train_configs/train.py --experiment fig3_diversity_16k --model-path /path/to/Qwen3-1.7B-Base
python verl/train_configs/train.py --experiment fig4_easy_444     --model-path /path/to/Qwen3-8B-Base
python verl/train_configs/train.py --experiment fig4_random_3k    --model-path /path/to/Qwen3-8B-Base
python verl/train_configs/train.py --experiment fig5_simple_ce    --model-path /path/to/Qwen3-4B-Base
python verl/train_configs/train.py --experiment fig5_simple_p     --model-path /path/to/Qwen3-4B-Base
python verl/train_configs/train.py --experiment fig5_medium_ce    --model-path /path/to/Qwen3-4B-Base
python verl/train_configs/train.py --experiment fig5_medium_p     --model-path /path/to/Qwen3-4B-Base
python verl/train_configs/train.py --experiment fig5_complex_ce   --model-path /path/to/Qwen3-4B-Base
python verl/train_configs/train.py --experiment fig5_complex_p    --model-path /path/to/Qwen3-4B-Base
python math_evaluation/sh/run_eval.py --experiment fig5_complex_p
```

If you prefer the original launcher style, the first command above expands to the following core `verl/` command (the Python launcher supplies the remaining options from `experiments.yaml`):

```bash
cd verl
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
  data.train_files=data/dapo17k_qwen3_4b_4k/train_pos.parquet data.val_files=null \
  data.prompt_key=extra_info data.response_key=extra_info \
  'data.prompt_dict_keys=[question]' 'data.response_dict_keys=[answer]' \
  data.train_batch_size=128 data.micro_batch_size_per_gpu=4 data.max_length=4800 \
  optim.lr=1e-5 model.partial_pretrain=/path/to/Qwen3-1.7B-Base \
  model.fsdp_config.model_dtype=bf16 model.use_liger=true \
  trainer.default_local_dir=checkpoints/fig1_correct_ce \
  trainer.logger='[console,tensorboard]' trainer.total_epochs=3 \
  trainer.test_freq=-1 trainer.save_freq=-1 use_remove_padding=true
```

For `-p`, switch to `-m verl.trainer.fsdp_dft_trainer`; for the incorrect set, switch to `data/dapo17k_qwen3_4b_4k/train_neg.parquet`. Prefer `train_configs/train.py` when using the complete parameterized settings. Checkpoints and TensorBoard logs go under ignored `verl/checkpoints/` and `verl/tensorboard_log/`; evaluation results go under ignored `math_evaluation/outputs/`. The Python launchers refuse to overwrite existing output directories. You may set `--gpus` for training and `--tensor-parallel-size` for evaluation; hardware changes may affect results.

The math evaluation runs AIME 2024, AMC23, Minerva Math, OlympiadBench and a fixed 2,000-question MATH subset at temperature 1.0 with 16 samples; `summary.json` reports `avg@16` averaged equally over the five benchmarks. **This MATH subset is a new seed-42 draw** because the original paper's 2,000 sampled IDs were not preserved here; it is not claimed to reproduce the paper's exact reported MATH number. The copied historical launcher evaluated 1,000 MATH questions, so it is intentionally not used as-is. The source NuminaMath launcher ran 1 epoch (as reflected in the simple-CoT config), whereas the paper states 3 epochs as its default: this discrepancy is disclosed rather than silently changed.

## Data release and attribution

The two ignored folders `verl/data/` and `math_evaluation/data/` (~0.6 GB combined) are intended to be uploaded **separately** to a Hugging Face Dataset, preserving these paths. After downloading them into the new repository, run `python verl/train_configs/verify_data.py`; the checksum manifest is kept with the training configs. To recreate the same files from an adjacent copy of the original repository, run `python verl/train_configs/prepare_data.py --source-root ../DFT` (or `--resume` to verify an existing copy). This requires no network. **No upload, download, or full training was performed here.**

The `verl/` copy retains the upstream Apache-2.0 `LICENSE` and `Notice.txt`; `math_evaluation/` and its `latex2sympy/` copy retain their MIT licenses. The NuminaMath data card lists Apache-2.0; the owner confirmed redistribution of the relevant problems and benchmarks. The `-p` objective is changed **only inside the new `verl/` copy**, and its gradient was checked against direct `-p`. Please review the files and licenses before publishing. These are selected experiments, not every model or domain in the paper.

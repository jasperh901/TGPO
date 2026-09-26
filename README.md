# TGPO

Training and evaluation code for Tracked-Gradient Policy Optimization.

[Experimental figures and tables](docs/results.md) · [Training](#3-train) · [Evaluation](#4-evaluate)
TGPO coordinates multiple reward objectives using temporally tracked policy gradients. It preserves GDPO's reward-decoupled normalization, maintains a bounded recursive estimate for each objective, and finds a conflict-averse direction near their mean. The same tracked vectors determine the interaction geometry and the gradient applied to the optimizer.

This repository provides the training runtimes, five training configurations, environment locks, data preparation, checkpoint export, and evaluation pipelines for mathematical reasoning, coding reasoning, and tool use. Datasets and pretrained models are downloaded from their original sources. Training outputs and weights are stored locally under ignored directories.

## Experimental results

The following endpoints summarize correctness within the response budget and format adherence. Rates are percentages; mean length is in tokens. The [complete result tables and six experimental figures](docs/results.md) cover both model scales, all mathematical benchmarks, code generation, tool use, parameter sensitivity, and mechanism ablation.

| Task / model | Endpoint | GDPO | DVAO | TGPO (Ours) |
| --- | --- | ---: | ---: | ---: |
| Math / 1.5B | Macro accuracy ↑ | 30.74 | 29.65 | **33.07** |
| Math / 1.5B | Overlength ↓ | 1.28 | 2.24 | **0.40** |
| Code / 1.5B | Correct and ≤4,000 tokens ↑ | 13.54 | 13.81 | **15.73** |
| Code / 1.5B | Mean tokens ↓ | 7298.08 | 6885.39 | **2666.17** |
| Tool / 1.5B | Strict format ↑ | 92.73 | 77.18 | **96.06** |
| Tool / 3B | Multi-turn format ↑ | 75.25 | 54.75 | **89.38** |

Baselines: [GDPO](https://github.com/NVlabs/GDPO) and [DVAO](https://arxiv.org/abs/2605.25604). Model and dataset sources are linked in [Download models and prepare datasets](#2-download-models-and-prepare-datasets).

**Mathematical training:** correctness reward, length reward, and mean response length.

![Mathematical training curves](docs/assets/figures/math_performance.webp)

**Coding training:** test-pass, syntax, and length rewards.

![Coding training curves](docs/assets/figures/code_performance.webp)

**Mechanism ablation:** full TGPO, no geometry, and no tracking on BFCL-v3 after 40 training steps.

![Tool-use mechanism ablation](docs/assets/figures/tool_ablation.webp)

Additional plots: [training dynamics](docs/assets/figures/math_training.webp), [parameter sensitivity](docs/assets/figures/math_parameter_sensitivity.webp), and [accuracy–length compliance](docs/assets/figures/math_pareto.webp).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Gradient tracking, coordination, FSDP caching, and tracker checkpoints |
| `runtime/` | Task-specific verl/GDPO runtimes used by the training configurations |
| `configs/` | Fully specified training configurations with portable paths |
| `env/` | Separate, pinned training and inference environments |
| `scripts/` | Downloads, preparation, training, export, and evaluation |
| `eval/bfcl/` | Adapter and patch for the pinned official BFCL-v3 evaluator |
| `tests/` | Gradient, optimizer, checkpoint, and distributed equivalence checks |
| `docs/` | Project website and public experimental figures and tables |

## 1. Install the environments

Use Linux, Python 3.10, NVIDIA GPUs, and a CUDA-compatible driver. Training uses PyTorch **2.4.0 + CUDA 12.1**, vLLM **0.6.3**, Transformers **4.47.1**, and FlashAttention **2.6.3**. Code generation and BFCL evaluation use a separate environment with PyTorch **2.6.0 + CUDA 12.4**, vLLM **0.8.5**, and Transformers **4.51.3**. Full dependency versions are in `env/*.lock`.

Download this repository as a ZIP archive and save it as `TGPO.zip`. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, a C/C++ toolchain, and CUDA development tools before running:

```bash
unzip TGPO.zip -d TGPO
cd TGPO
bash env/setup_train.sh
bash env/setup_eval.sh
```

The scripts create `.venv-train` and `.venv-eval`. FlashAttention is compiled against the training environment; to use an existing compatible wheel, set `TGPO_FLASH_WHEEL=/path/to/flash_attn.whl`. `TGPO_TRAIN_ENV` and `TGPO_EVAL_ENV` select alternative environment locations.

Code training and scoring also require **bubblewrap** and **util-linux** with unprivileged user namespaces enabled. On Debian/Ubuntu, install them with `sudo apt-get install bubblewrap util-linux`. Verify the isolated judge before a run:

```bash
.venv-train/bin/python scripts/score_code.py --smoke-test
```

This executes a passing program and a program that fails its eleventh test. Generated programs run in a filesystem and network sandbox.

## 2. Download models and prepare datasets

Run preparation with the training environment. Models are saved under `models/`, datasets under `data/`, and resolved model revisions are recorded during download.

```bash
# Download all four models used by the five configurations.
.venv-train/bin/python scripts/download_models.py all
.venv-train/bin/python scripts/download_data.py all
```

To reproduce one task at a time:

```bash
# Mathematical reasoning, 1.5B
.venv-train/bin/python scripts/download_models.py math_1.5b
.venv-train/bin/python scripts/download_data.py math

# Coding reasoning, 1.5B: preprocessing uses the original 7B tokenizer.
.venv-train/bin/python scripts/download_models.py code_1.5b
.venv-train/bin/python scripts/download_models.py math_7b --tokenizer-only
.venv-train/bin/python scripts/download_data.py code

# Tool use, 1.5B; select tool_3b for the 3B model.
.venv-train/bin/python scripts/download_models.py tool_1.5b
.venv-train/bin/python scripts/download_data.py tool
```

For 7B math alone, also download the 1.5B tokenizer with `scripts/download_models.py math_1.5b --tokenizer-only`; shared math preparation uses it to record prompt lengths. `--data-dir` and `--model-dir` override default locations in the download and training scripts.

### Models

| Profiles | Pretrained model |
| --- | --- |
| `math_1.5b`, `code_1.5b` | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| `math_7b` | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| `tool_1.5b` | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| `tool_3b` | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |

### Data sources and preparation

| Task | Training source | Evaluation source | Preparation |
| --- | --- | --- | --- |
| Math | [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset), revision `b6ae8c6` | MATH-500, AIME 2024, AMC 2022–2023, Minerva Math, and OlympiadBench from the [PRIME mirror](https://github.com/PRIME-RL/PRIME/tree/18ad596f08d487bb546d80d738d99ec697bd2e75/eval/data) | Preserve all 40,315 training records and use the GDPO-style math instruction. Evaluation has 1,560 questions. |
| Code | [Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data), revision `9776b13` | Validation split: APPS, CodeContests, Codeforces, and TACO | Select `ability=code`; retain prompts of at most 1,024 tokens using the DeepSeek-R1-Distill-Qwen-7B tokenizer: 25,092 training and 1,016 validation records. |
| Tool | [ToolRL rlla_4k](https://github.com/qiancheng0/ToolRL/tree/8cee13ec0ca72f0461da372a93a6fd8140dbb840/dataset/rlla_4k) | [BFCL-v3](https://github.com/ShishirPatil/gorilla/tree/ea13468e4423454d0c213704fb87cf7cb3990433/berkeley-function-call-leaderboard) | Preserve ToolRL's prepared messages and reward fields; fetch the pinned BFCL evaluator and apply the supplied adapter. |

The downloader pins data snapshots and checks SHA-256 digests for DeepScaleR, math evaluation files, and ToolRL. Code preprocessing checks retained split sizes and records source counts. Original dataset and model licenses apply to downloads; consult the linked repositories and model cards.

## 3. Train

The profiles below provide the training settings. Memory requirements depend on model size, sequence length, FSDP settings, and the number of tracked objectives. The 7B math profile is intended for two A800-class GPUs; the other profiles use eight GPUs.

| Profile | Outer steps | Actor updates | Prompts / rollouts per prompt | Response cap | Objectives |
| --- | ---: | ---: | --- | ---: | --- |
| `math_1.5b` | 500 | 500 | 16 / 8 | 2,048 | Correctness, length |
| `math_7b` | 500 | 500 | 16 / 8 | 2,048 | Correctness, length |
| `code_1.5b` | 400 | 400 | 8 / 8 | 5,120 | Correctness, syntax, length |
| `tool_1.5b` | 100 | 400 | 512 / 4 | 1,024 | Correctness, format |
| `tool_3b` | 100 | 400 | 512 / 4 | 1,024 | Correctness, format |

All profiles use tracker bound `R=1`, coordination radius `rho=0.125`, a resolution-100 simplex grid, learning rate `1e-6`, and tracking coefficient `1/sqrt(T)`, where `T` is the **actor-update** horizon. Length targets are 1,024 tokens for math and 4,000 for code.

```bash
.venv-train/bin/python scripts/train.py math_1.5b --gpus 0,1,2,3,4,5,6,7
.venv-train/bin/python scripts/train.py math_7b   --gpus 0,1
.venv-train/bin/python scripts/train.py code_1.5b --gpus 0,1,2,3,4,5,6,7
.venv-train/bin/python scripts/train.py tool_1.5b --gpus 0,1,2,3,4,5,6,7
.venv-train/bin/python scripts/train.py tool_3b   --gpus 0,1,2,3,4,5,6,7
```

Run each command as a separate experiment. The launcher selects the task runtime, resolves paths, saves the effective configuration, and writes logs and checkpoints under `outputs/<profile>_tgpo/`. Use `--output` for another destination. Inspect any configuration without allocating GPUs using `--dry-run`.

The same launcher exposes comparison methods:

```bash
.venv-train/bin/python scripts/train.py math_1.5b --algorithm gdpo --gpus 0,1,2,3,4,5,6,7
.venv-train/bin/python scripts/train.py math_1.5b --algorithm dvao --gpus 0,1,2,3,4,5,6,7
```

Change parameters with repeatable `--override key=value` arguments:

```bash
.venv-train/bin/python scripts/train.py math_1.5b --dry-run \
  --override actor_rollout_ref.actor.moco_cagrad_shadow.radius=0.25
```

When changing duration or minibatch scheduling, set `horizon_actor_updates` to the planned number of actor optimizer steps. The fully resolved YAML defines the experiment.

### Checkpoints

Math 1.5B, code, and tool profiles save Hugging Face actor checkpoints at `outputs/<profile>_tgpo/checkpoints/actor/global_step_<step>/`. Math 7B saves a complete snapshot at `outputs/math_7b_tgpo/checkpoints/latest/`, including the actor, optimizer, RNG, and tracker states. This directory also serves as its evaluation checkpoint.

## 4. Evaluate

### Mathematical reasoning

Use the training environment. The evaluator generates four responses per question with seeds `0,1,2,3`, temperature `0.6`, top-p `0.95`, and a 2,048-token cap, then grades correctness and compliance with the 1,024-token target.

```bash
.venv-train/bin/python scripts/eval_math.py \
  --checkpoint outputs/math_1.5b_tgpo/checkpoints/actor/global_step_500 \
  --output outputs/eval_math_1.5b_tgpo --gpus 0,1,2,3
```

For 7B, use `outputs/math_7b_tgpo/checkpoints/latest`. Evaluation covers 500 MATH, 30 AIME, 83 AMC, 272 Minerva, and 675 OlympiadBench questions. Grading retries symbolic timeouts and preserves sampling identities. `--dry-run` prints the generation and scoring commands.

### Coding reasoning

Generate with the evaluation environment, then score with the training environment:

```bash
.venv-eval/bin/python scripts/eval_code.py \
  --checkpoint outputs/code_1.5b_tgpo/checkpoints/actor/global_step_400 \
  --output outputs/eval_code_1.5b_tgpo --gpus 0,1,2,3,4,5,6,7

.venv-train/bin/python scripts/score_code.py \
  --generation-dir outputs/eval_code_1.5b_tgpo --workers 32
```

Evaluation uses four responses per question, temperature `0.6`, top-p `0.95`, and a 32,768-token cap. Base seeds are `2026091601` through `2026091604`; deterministic hashing gives each question/sample its request seed. The 4,000-token target is assessed separately from truncation. Every available unit test is executed, with a five-second limit per test. Two malformed validation records, at zero-based positions 18 and 958, are excluded consistently, giving **1,014 scored questions** and **4,056 responses**. Reports include pass@1, correctness within the length target, overlength rate, syntax validity, and mean response length.

### Tool use

`env/setup_eval.sh` fetches pinned BFCL-v3 and applies `eval/bfcl/adapter.patch`. The adapter retains ToolRL's message format, parser, and strict-format checks. First export a clean inference checkpoint:

```bash
.venv-train/bin/python scripts/export_checkpoint.py \
  --source outputs/tool_1.5b_tgpo/checkpoints/actor/global_step_100 \
  --output outputs/tool_1.5b_tgpo/export

.venv-eval/bin/python scripts/eval_tool.py \
  --checkpoint outputs/tool_1.5b_tgpo/export \
  --output-root outputs/eval_tool_1.5b_tgpo --gpus 0,1,2,3,4,5,6,7

.venv-eval/bin/python scripts/score_tool.py \
  --run-root outputs/eval_tool_1.5b_tgpo
```

Use corresponding `tool_3b` paths for 3B. Export runs on CPU and needs host memory for the model. BFCL evaluation uses seed `20260826`, temperature `0.001`, and top-p `1.0`. Overall accuracy averages non-live, live, and multi-turn accuracy. Strict format is response-weighted; a multi-turn trajectory must emit at least one response and every response must be structurally valid. The official scorer and format summary are retained in the output directory. For a quick integration run, use `--test-category simple --limit-per-category 2` before full evaluation.

## Implementation and verification

`src/moco_cagrad_shadow.py` implements the projected tracker and conflict-averse coordination. `src/moco_checkpoint.py` serializes tracker history. Task actors in `runtime/*/verl/workers/actor/dp_actor.py` compute objective gradients and apply coordinated updates. Internal names such as `moco_cagrad_shadow` and `scppo_shadow` preserve configuration and checkpoint compatibility; the public method is **TGPO**.

```bash
PYTHONPATH=src .venv-train/bin/python -m unittest discover -s tests -p 'test_moco*.py' -v
PYTHONPATH=src .venv-train/bin/torchrun --standalone --nproc-per-node=2 tests/test_distributed_moco.py
PYTHONPATH=src .venv-train/bin/torchrun --standalone --nproc-per-node=2 tests/test_distributed_moco_cache.py
```

These CPU tests compare updates with an independent vector reference, verify optimizer and RNG preservation, check exact checkpoint restoration, and exercise distributed reductions and FSDP caching. Full training and benchmark inference require the GPU resources above.

## Acknowledgments and licenses

TGPO builds on [GDPO](https://github.com/NVlabs/GDPO), [verl](https://github.com/volcengine/verl), [PRIME](https://github.com/PRIME-RL/PRIME), [ToolRL](https://github.com/qiancheng0/ToolRL), and [BFCL](https://github.com/ShishirPatil/gorilla). See [NOTICE](NOTICE) for source snapshots and [LICENSE](LICENSE) for the Apache-2.0 code license. Upstream notices remain in source files.

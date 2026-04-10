# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for the paper **"Context Tuning for In-Context Optimization"**. Context Tuning (CT) optimizes the KV cache produced by in-context demonstrations via gradient descent at test time, using a leave-one-out objective. The paper was rejected from its first venue and is currently under review at a new venue (Mar 2026 rebuttal in progress).

## Environment

uv venv at `.venv/` with Python 3.12. Key packages:
- `torch==2.4.0+cu121`, `transformers==4.47.1`, `accelerate==1.1.1`
- `peft==0.18.1` (upgraded from 0.13.2 for `RandLoraConfig`, `C3AConfig`, `LNTuningConfig`)
- `datasets==4.8.4` (upgraded from 3.1.0 — old version had dataclass bug on Python 3.12)
- `bitsandbytes==0.44.0`, `flash-attn==2.7.2.post1` (opt-in via `--flash_attn`)

Always use `.venv/bin/python` or `.venv/bin/accelerate` — never bare `python` or `source activate`.

## Rebuttal Scope

**3 benchmarks only: BBH, MMLU, ARC.** NLP-LR is excluded — the fine-tuned GPT-2 weights are lost.

## Running Experiments

**Local smoke test (on GPU node):**
```bash
# BBH ICL — ~4 min, uses ~32GB VRAM
.venv/bin/accelerate launch --main_process_port 29500 --mixed_precision bf16 \
    inference_bbh/test_time_evaluate.py --tag smoke_bbh_icl --seed 42 --eval_ratio 0.01

# BBH CT-KV (minimal) — ~4 min
.venv/bin/accelerate launch --main_process_port 29500 --mixed_precision bf16 \
    inference_bbh/test_time_evaluate.py --tag smoke_bbh_ctkv --seed 42 --eval_ratio 0.01 \
    --gs_epochs 2 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# MMLU ICL — ~2 min
.venv/bin/accelerate launch --main_process_port 29501 --mixed_precision bf16 \
    inference_mmlu/test_time_evaluate.py --tag smoke_mmlu_icl --seed 42 --eval_ratio 0.01

# ARC ICL (part1, ~80 tasks) — ~3 min, no --eval_ratio (not supported)
.venv/bin/accelerate launch --main_process_port 29502 --mixed_precision bf16 \
    inference_arc/test_time_evaluate.py --tag smoke_arc_icl \
    --select_tasks_path data/task_info_part1.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24
```

**Colocating smoke tests:** H200 (141GB) can run 2 concurrent tests safely (~32GB each). 3 concurrent may OOM. Use different `--main_process_port` values.

**Batch submission (NEW format — use for all new experiments):**
```bash
makesbatch --time 6 --ngpu 1 --gb 64 --bash_file bash_cmds/0326_1_experiment/bbh.sh
```
See `notes/paper.md` Section 12 for the new bash_cmds format rules.

**Results** land in `encoder_decoder/outputs_eval/{tag}/` with `eval_pred_gt.json`. Parse accuracy with `random_utils/get_exact_acc.py`.

## Architecture

### Per-Benchmark Directories (the code is duplicated, not shared)

Each `inference_*` directory is self-contained with benchmark-specific variants of the same files:

| File | Purpose |
|------|---------|
| `test_time_evaluate.py` | Central script (~2300-2800 lines): TTT, KV init, gradient search (CT-KV), evaluation loop |
| `custom_llama.py` | Modified `LlamaForCausalLM` with pre-RoPE key extraction and position-ID subtraction hack |
| `data_utils.py` | `EvalDataset`, `GSDataset` (KV optimization pairs), `TTTDataset` (LoRA training permutations) |
| `tasks.py` | Task definitions (BBH only) |

**Benchmark → Model mapping:**

| Directory | Benchmark | Base Model | Extra Args |
|-----------|-----------|------------|------------|
| `inference_bbh/` | BBH (23 tasks) | Llama-3.2-1B-Instruct | `--seed {42..46}` |
| `inference_mmlu/` | MMLU (57 subjects) | Llama-3.2-1B-Instruct | `--seed {42..46}` |
| `inference_arc/` | ARC (400 tasks) | Llama-3.2-1B-Instruct (fine-tuned) | `--select_tasks_path data/task_info_part{1..5}.csv --no_bos --weight_dir 0317_noprogram_base --weight_epoch 24` |
| `inference_nlp/` | NLP-LR (MetaICL) | GPT-2 Large | **OUT OF SCOPE** — fine-tuned weights lost |
| `inference_*_bigllm/` | BBH/MMLU with larger models | Qwen/DeepSeek/Mistral 12-32B | `--model_name qwen14b --untrainable_nbit 4` |
| `inference_*_prompt/` | CT-Prompt variant | Same as parent | Optimizes embeddings instead of KV |

### Method Pipeline (inside `test_time_evaluate.py`)

Per task: `deep-copy model` → `[optional TTT: LoRA fine-tune on demos]` → `initialize KV cache from demos` → `[optional CT-KV: gradient-optimize KV]` → `generate answers with KV prepended`

Key functions in `test_time_evaluate.py`:
- `run_ttt()` — LoRA adaptation on permuted demo pairs (~line 1100-1300)
- `initialize_kv()` — KV cache from demos, supports permutation averaging (~line 200-400)
- `run_gs()` — CT-KV gradient search with leave-one-out + token dropout (~line 1600-2130)
- `test_time_evaluate()` — orchestrates per-task adaptation + per-example generation (~line 500-1100)
- `main()` — argparse + model loading (~line 2360+)

### `custom_llama.py` — Two Critical Modifications

1. **`key_states_no_pos`**: Attention layers return keys BEFORE RoPE, enabling permutation-average-then-reapply-position.
2. **`model.subtract_position_ids_by`**: Corrects position IDs during generation when KV cache length differs from original demo token length.

### Data Flow

- `EvalDataset`: Loads tasks, formats `[instruction][demo1_Q][demo1_A]...[test_Q]`, validates all examples of a task share identical `demon_input_ids`.
- `GSDataset`: One entry per demo pair for KV optimization. `example_idx` tracks which demo → used for leave-one-out masking.
- `TTTDataset`: Random permutations of demos, last pair is leave-one-out test. Loss on answer tokens only.

## bash_cmds Format (NEW — for all new experiments)

Directory naming: `bash_cmds/MMDD_N_description/`. File format:
```bash
# Experiment description
# makesbatch --time 6 --ngpu 1 --gb 64 --bash_file bash_cmds/MMDD_N_desc/file.sh

# globally_unique_job_name
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag globally_unique_job_name \
    --model_name llama1b \
    --gs_epochs 16 --gs_lr 1e-3
```

**Rules:** (1) strict 1:1 `# job_name` to command, (2) `--tag` matches job name exactly, (3) job names globally unique across ALL `.sh` files, (4) no `$MASTER_PORT` in commands, (5) use `.venv/bin/accelerate launch`, (6) `#!` tracking lines at bottom are managed by makesbatch.

## Key Hyperparameters (Best Configs)

| | CT-KV lr | CT-KV iters | tokdrop | TTT lr | TTT iters | TTT+CT-KV iters |
|-|----------|------------|---------|--------|-----------|-----------------|
| BBH | 1e-3 | 16 | 0.1 | 1e-4 | 8 | 8 |
| MMLU | 1.5e-3 | 20 | 0.1 | 1e-4 | 20 | 10 |
| ARC | 3e-3 | 200 | 0.1 | 1e-4 | 200 | 50 |

All CT-KV configs use `--gs_dropout train` (leave-one-out). LR >= 1e-2 consistently fails.

## Important Paths

- **Model cache**: `encoder_decoder_cache/` (HuggingFace downloads)
- **Eval outputs**: `encoder_decoder/outputs_eval/{tag}/`
- **TTT checkpoints**: `encoder_decoder/outputs_ttt/` or saved via `--ttt_save` into `outputs_eval/`
- **ARC base model**: `encoder_decoder/outputs/0317_noprogram_base/` (LoRA checkpoint, epoch 24)
- **Datasets**: `data/BIG-Bench-Hard/`, `data/re-arc/`, `data/MetaICL/`, MMLU via HuggingFace `datasets`
- **ARC task splits**: `data/task_info_part{1..5}.csv`
- **Hyperparameter search notes**: `bash_cmds/{benchmark}/search_note.txt`
- **Detailed paper notes**: `notes/paper.md`

## Code Duplication Warning

The `inference_*` directories contain duplicated-but-divergent copies of the same core logic. Changes to one benchmark's `test_time_evaluate.py` must be manually ported to others if the feature should be shared. The `_prompt`, `_bigllm`, and `prefix_tuning_mlp/` variants are further copies. Always check which directory's script a bash command actually invokes.

**DO NOT REFACTOR.** The codebase is spaghetti and there are many obvious improvements, but the rebuttal deadline is <1 week. Any structural refactoring risks introducing regressions in a codebase with no test suite, where correctness bugs silently produce wrong numbers. Make minimal, targeted changes to the specific files needed for new experiments. Copy-paste between benchmark directories is acceptable and expected.


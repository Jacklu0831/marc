# Context Tuning for In-Context Optimization — Paper & Codebase Notes

## 1. Core Idea

**Context Tuning (CT)** treats the KV cache produced by in-context demonstrations as a set of optimizable parameters and refines them via gradient descent at test time. The key insight: standard ICL computes a single forward pass through demonstrations to produce a KV cache, but this representation is suboptimal. By optimizing the KV cache entries directly on a leave-one-out objective over the demonstrations, the model's internal representation of the task can be significantly improved.

Two main variants:
- **CT-KV**: Directly optimizes the KV cache entries (keys and/or values) across transformer layers.
- **CT-Prompt (CT-P)**: Optimizes soft prompt embeddings instead of KV entries (analogous to prompt tuning but with the leave-one-out training objective).

Both can be combined with **Test-Time Training (TTT)**, where LoRA adapters are first trained on demonstrations, then the resulting KV cache is further refined via CT-KV. The combination **TTT + CT-KV** achieves the best results.

## 2. Method Details

### 2.1. Pipeline (Per-Task)

For a task with N demonstration pairs and M test examples:

1. **[Optional] TTT (LoRA fine-tuning)**: Generate `ttt_permute_n` unique permutations of the N demos. Each permutation uses N-1 as context and 1 as leave-one-out test. Train LoRA adapters (rank 64-128, targeting q/v/o/gate/up/down projections) for `ttt_iters` steps with next-token prediction loss on answer tokens. After training, merge LoRA into base model weights.

2. **KV Cache Initialization**: Run the (possibly TTT-adapted) model on concatenated demonstration tokens to produce an initial KV cache. Multiple initialization strategies exist:
   - **Standard**: Single forward pass (baseline ICL).
   - **Permutation Averaging**: Run multiple permutations of demo order, optionally strip positional encoding from keys before averaging, then re-apply RoPE in canonical order. This reduces position-dependent artifacts.
   - **Random KV**: Initialize from uniform distribution or model outputs on dummy tokens.
   - **Deep Thinking (DT)**: Iterative refinement — pass input through the model multiple times with existing KV cache, using EMA updates.

3. **[Optional] CT-KV (Gradient Search)**: Make KV cache entries trainable. For each epoch, iterate over demonstration pairs:
   - **Leave-one-out dropout** (`gs_dropout='train'`): When training on pair i, mask out pair i's KV entries so the model can't directly copy its own representation.
   - **Token dropout** (`gs_token_dropout`): Randomly drop individual tokens from the KV attention mask (Bernoulli mask, probability = token_dropout).
   - Forward pass with KV cache prepended, compute cross-entropy loss on answer tokens.
   - Accumulate gradients over all pairs, update once per epoch (AdamW, cosine LR schedule).
   - Optional: L2 regularization toward initial KV, Fisher information weighting, concurrent LoRA training.

4. **Evaluation**: For each test example, prepend the optimized KV cache and generate greedily. For ARC, use D8 geometric augmentation + majority voting.

### 2.2. Custom Llama Modifications (`custom_llama.py`)

Two critical modifications to HuggingFace's `LlamaForCausalLM`:

1. **`key_states_no_pos`**: Each attention layer can return key states BEFORE RoPE (rotary position embedding) is applied. This is essential for permutation averaging — keys from different permutations have different positional encodings, so you must strip position before averaging and re-apply afterward.

2. **`subtract_position_ids_by`**: During generation, HuggingFace auto-computes position_ids starting from `past_key_values_length`. When the KV cache length differs from the original demo length (e.g., with `permute_concat`), position IDs would be wrong. This field subtracts the difference so generation positions continue correctly.

### 2.3. Leave-One-Out Training (The Key Regularization)

The leave-one-out mechanism is the primary technical contribution for preventing overfitting:
- When optimizing the KV cache on demonstration pair i, the model cannot attend to the KV entries that were produced from pair i's tokens.
- This forces the optimized KV to capture generalizable task structure rather than memorizing specific demo pairs.
- Implemented via attention masking (non-detach mode) or gradient detachment (detach mode).

Five dropout strategies are implemented: `none`, `train` (leave-one-out), `suffix`, `power` (random subset), `power_with_train`.

### 2.4. Token Dropout (Secondary Regularization)

After structural leave-one-out dropout, individual tokens in the KV cache are randomly masked with probability `token_dropout`. This acts as a form of dropout on the context representation, preventing reliance on specific token patterns.

## 3. Benchmarks and Models

| Benchmark | Type | Tasks | Base Model | Key Details |
|-----------|------|-------|------------|-------------|
| **ARC** | Visual pattern reasoning (grids) | 400 eval tasks | Llama-3.2-1B-Instruct (fine-tuned on re-arc) | Custom tokenizer for grids; D8 augmentation + voting |
| **BBH** | BIG-Bench Hard reasoning | 23 tasks | Llama-3.2-1B-Instruct (off-the-shelf) | 27 original tasks, 23 used |
| **MMLU** | Multi-domain knowledge QA | 57 subjects | Llama-3.2-1B-Instruct (off-the-shelf) | eval_ratio varies (0.25 to 1.0) |
| **NLP-LR** | NLP classification (MetaICL) | 21 task families, 5 seeds | GPT-2 Large (off-the-shelf) | Low-resource transfer tasks |

For ARC, a base model is first fine-tuned on re-arc synthetic training data with LoRA, checkpoint `0317_noprogram_base` epoch 24. All other benchmarks use the model off-the-shelf.

Rebuttal experiments also test larger models: Mistral 12B, Qwen 14B/32B, DeepSeek 14B/32B on BBH and MMLU.

## 4. Selected Hyperparameters (Best Configs)

| Benchmark | CT-KV | CT-P | Prefix Tuning (m=32) | TTT | TTT+CT-KV |
|-----------|-------|------|---------------------|-----|-----------|
| **ARC** | lr=3e-3, iter=200, tokdrop=0.1 | lr=1e-3, iter=150, tokdrop=0.1 | lr=3e-3, iter=150 | lr=1e-4, iter=200 | iter=50 |
| **NLP-LR** | lr=1e-3, iter=250, tokdrop=0.05 | lr=1e-3, iter=250, tokdrop=0.05 | lr=3e-3, iter=250 | lr=1e-4, iter=250 | iter=30 |
| **BBH** | lr=1e-3, iter=16, tokdrop=0.1 | lr=3e-4, iter=12 | lr=3e-4, iter=16 | lr=1e-4, iter=8 | iter=8 |
| **MMLU** | lr=1e-3, iter=20, tokdrop=0.1 | lr=3e-4, iter=20 | lr=1e-3, iter=20 | lr=1e-4, iter=20 | iter=10 |

Notes:
- Token dropout applies to all methods except TTT.
- CT-KV and prefix/KV-tuning need higher learning rates (1e-3 to 3e-3) because the KV entries are "not relevant" initially (need bigger steps).
- TTT always uses lr=1e-4 (standard LoRA fine-tuning rate).
- TTT+CT-KV runs fewer CT-KV iterations (the TTT-adapted KV is already a better starting point).

## 5. Main Paper Results

### 5.1. NLP-LR Accuracy vs. Training Time (Main Figure)

| Method | Accuracy (%) | Time/Task (sec) |
|--------|-------------|-----------------|
| Zero-shot | 34.9 | 0 |
| In-Context Learning | 35.6 | 0 |
| Prompt-Tuning | 41.4 | 147 |
| Prefix-Tuning (m=32) | 42.0 | 123 |
| Test-Time Training | 44.1 | 342 |
| **CT-Prompt** | **43.2** | **228** |
| **CT-KV** | **44.4** | **145** |
| **TTT + CT-KV** | **47.3** | **372** |

Key claim: CT-KV matches TTT performance (44.4 vs 44.1) at less than half the compute (145s vs 342s). TTT + CT-KV beats all methods by 3+ points.

### 5.2. Ablation (Leave-One-Out + Token Dropout)

| Config | NLP-LR | BBH | MMLU | ARC |
|--------|--------|-----|------|-----|
| Neither | 41.0 | 51.4 | 40.2 | 21.0 |
| + Leave-one-out only | 43.6 | 54.4 | 41.5 | 23.8 |
| + Token-dropout only | 43.9 | 55.3 | 42.7 | 21.0 |
| Both | 44.2 | 57.9 | 43.7 | 22.5 |

Both regularizations contribute significantly. Largest gains on BBH (+6.5 from neither to both). ARC is anomalous — token dropout alone doesn't help, but leave-one-out gives the biggest single boost.

### 5.3. Eval on Demonstrations (Memorization Test)

| Benchmark | Accuracy on Training Demos |
|-----------|---------------------------|
| ARC | 23% |
| NLP-LR | 82% |
| BBH | 84% |
| MMLU | 89-93% |

Shows the optimized KV effectively encodes the task. ARC's low score reflects the benchmark's difficulty even with memorized context.

### 5.4. ARC Task-Level Analysis

- 44 tasks solved by both ICL and CT-KV
- 9 solved only by ICL (regression)
- 51 solved only by CT-KV (new)
- 296 unsolved by both
- ICL baseline: ~13.25%, CT-KV: ~23.75% (nearly doubled solved tasks)

## 6. Previous Rebuttal Experiments (Feb 2026 — REJECTED)

The paper was rejected from its first venue. The rebuttal (commit `1ced20b`) added experiments addressing that round's reviewer concerns. These are documented here for reference but are from the **failed** rebuttal:

### 6.1. Scaling with Number of Demonstrations

NLP-LR (k = 8, 16, 24, 32):
- ICL: flat ~36-37%
- Prefix Tuning: 40.5% -> 44.0%
- CT-KV: 43.1% -> 48.9% (largest gain, +5.8 over range)

MMLU (k = 16, 32, 48, 64):
- ICL: flat ~43-44%
- Prefix Tuning: 42.4% -> 44.2%
- CT-KV: 45.5% -> 47.6% (consistently best)

**Takeaway**: CT-KV benefits most from more demonstrations, widening its advantage.

### 6.2. Robustness to Label Corruption

NLP-LR (corruption p = 0%, 25%, 50%, 75%, 100%):
- ICL: 35.6 -> 31.7 (relatively flat, barely uses labels)
- Prefix Tuning: 42.0 -> 31.4 (steep drop)
- CT-KV: 44.2 -> 31.4 (similar drop rate to prefix but starts higher)

MMLU (corruption p = 0%, 25%, 50%, 75%, 100%):
- ICL: 41.2 -> 40.2 (very flat)
- Prefix Tuning: 39.9 -> 37.4
- CT-KV: 43.7 -> 39.1 (degrades more gracefully than prefix)

**Takeaway**: CT-KV degrades proportionally to corruption level but maintains advantage over prefix tuning across all corruption rates. At full corruption (100%), CT-KV and Prefix converge to similar performance, suggesting the method does rely on labels but not excessively.

### 6.3. Larger Models (Big-LLM Experiments)

Experiments on BBH and MMLU with Mistral-12B, Qwen-14B/32B, DeepSeek-14B/32B comparing CT-KV against Prefix Tuning. These address the "does it generalize beyond 1B models?" concern.

### 6.4. Additional PEFT Variants

The rebuttal code adds support for:
- **DoRA** (Weight-Decomposed Low-Rank Adaptation)
- **RandLoRA** (Random Low-Rank Adaptation)
- **LN-Tuning** (LayerNorm Tuning)
- **C3A** (Cross-Covariance Adapter)
- **Full fine-tuning** (all model parameters)

These can be used alongside CT-KV optimization (joint KV + adapter tuning) to compare against more PEFT baselines.

## 7. Codebase Architecture

### 7.1. Directory Structure

```
marc/
  inference_{benchmark}/          # Main code for each benchmark
    test_time_evaluate.py         # Central evaluation script (~2300 lines)
    custom_llama.py               # Modified Llama with key_states_no_pos
    data_utils.py                 # Datasets: EvalDataset, GSDataset, TTTDataset
    train.py                      # Base model fine-tuning (ARC only meaningful)
    tasks.py                      # Task definitions (BBH only)
  inference_{benchmark}_prompt/   # CT-Prompt variant (optimizes embeddings not KV)
  inference_{benchmark}_bigllm/   # Larger model experiments (BBH, MMLU)
  prefix_tuning_mlp/              # MLP-based prefix tuning variant
  encoder_decoder/                # Model checkpoints and eval outputs
  encoder_decoder_cache/          # HuggingFace model cache
  data/                           # Datasets (BIG-Bench-Hard, ConceptARC, MetaICL, re-arc)
  bash_cmds/{arc,bbh,mmlu,nlp}/   # Experiment scripts organized by benchmark
  neurips_submission/plots/       # Paper figures and plotting scripts
  sbatch_files/                   # Generated SLURM files (gitignored)
  slurm_outs/                     # SLURM logs (gitignored)
```

### 7.2. Code Duplication

Each `inference_*` directory is largely self-contained with its own copies of `test_time_evaluate.py`, `custom_llama.py`, and `data_utils.py`. These files are similar but have benchmark-specific differences (e.g., ARC has augmentation/voting, NLP uses GPT-2, BBH/MMLU use Llama). There is substantial code duplication across directories — this is acknowledged as "scrappy."

### 7.3. Key Execution Pattern

```bash
# Training (ARC only):
accelerate launch inference_arc/train.py --weight_dir ... --epochs ...

# Evaluation with CT-KV:
accelerate launch inference_bbh/test_time_evaluate.py \
    --model_name llama1b \
    --gs_epochs 16 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --tag my_experiment

# Evaluation with TTT + CT-KV:
accelerate launch inference_bbh/test_time_evaluate.py \
    --model_name llama1b \
    --ttt_iters 8 --ttt_lr 1e-4 \
    --gs_epochs 8 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --tag my_ttt_kv_experiment
```

### 7.4. Output Structure

Results saved to `encoder_decoder/outputs_eval/{tag}/` containing:
- `results.json` with per-task accuracy
- Various metadata

## 8. Important Implementation Details

### 8.1. Gradient Accumulation in GS

Gradients accumulate across ALL batches within an epoch. `optim.step()` and `scheduler.step()` are called once per epoch (end), not per batch. This means the effective batch size equals the total number of demonstration pairs per epoch.

### 8.2. Position ID Management

The interaction between KV cache length and position IDs is tricky:
- Standard ICL: position IDs during generation start at `len(demon_tokens)`, which equals `len(KV cache)`.
- Permute-concat: KV length = `num_permute * len(demon_tokens)`, but positions should still start at `len(demon_tokens)`. The `subtract_position_ids_by` hack corrects this.
- Zero-shot: KV is truncated to instruction-only portion; no subtraction needed.

### 8.3. ARC-Specific Details

- Custom `ARCTokenizer` encodes grids as token sequences: `[dims] [row1_values] [\n] [row2_values] ...`
- D8 augmentations: 8 geometric transformations (identity, 3 rotations, horizontal flip, vertical flip, transpose, anti-transpose)
- Color permutation augmentation during training
- Majority voting across augmented predictions for final answer
- Base model is fine-tuned on re-arc (synthetic ARC-like tasks), checkpoint `0317_noprogram_base` at epoch 24

### 8.4. NLP-LR Specifics

- Uses GPT-2 Large (not Llama) as base model
- 21 task families from MetaICL, 5 random seeds
- Each "task" is a random seed × task family combination
- Standard ICL barely helps over zero-shot (~35.6 vs 34.9)

### 8.5. BBH/MMLU Specifics

- Both use Llama-3.2-1B-Instruct off-the-shelf (no fine-tuning)
- BBH: 23 tasks with structured prompts and answer formats
- MMLU: 57 subjects, can subsample test set via `eval_ratio`
- Both use many fewer optimization iterations than ARC/NLP (8-20 vs 150-250)

## 9. Key Observations from Search Notes

- **Learning rate sensitivity**: KV optimization needs higher LR (1e-3 to 3e-3) than TTT (1e-4). LR >= 1e-2 consistently fails ("sucks").
- **Token dropout sweet spot**: 0.05-0.1 works best; 0.2 sometimes helps but often hurts.
- **Leave-one-out is essential**: `gs_dropout='train'` used for all final configs.
- **TTT+CT-KV synergy**: After TTT, only a few CT-KV iterations are needed (8-50 vs 16-250 standalone). The TTT-adapted KV is a much better starting point.
- **Permutation averaging**: Tested but not used in final configs (standard single forward pass is the default for KV init).
- **Fisher regularization**: Computed but not used in final configs (lambda=0 best everywhere).
- **No-key ablation**: Optimizing only values (no keys) works but slightly worse than both.
- **Seed sensitivity**: MMLU results vary across seeds; seed 42-46 range chosen for reporting.
- **Memory constraints**: BBH TTT+CT-KV requires 48GB VRAM; MMLU seed 46 OOMs.

## 10. What's Potentially Deprecated / Scrappy

- `prefix_tuning_mlp/` — MLP-based prefix tuning variant, appears to be an abandoned direction
- `inference_*_prompt/` directories — CT-Prompt variant (CT-P), less prominent than CT-KV
- Many bash_cmds subdirectories for intermediate experiments: `3_randomsearchntoken/`, `4_randomsearchfull/`, `11_fisher/`, `progression/`, `numparams/`, `randommlpinit/` — these are hyperparameter search artifacts
- `data/re-arc/` contains the full DSL for generating ARC tasks, only used for training data
- `data/ConceptARC/` — explored but may not be in final paper
- The `encoder_decoder/` naming is vestigial (originally an encoder-decoder architecture was planned?)
- Multiple "deprecated" sections in search_note.txt files indicate iterative refinement

## 11. Submission Timeline

- **Nov 2024**: Initial commit, early development with encoder-decoder concepts, prefix tuning
- **Apr 2025**: Core method development (gradient search, TTT, Fisher, grid searches)
- **May 2025**: NeurIPS 2025 submission deadline (likely)
- **Jul 2025**: Camera-ready-level work (big-LLM experiments, PEFT variations, code cleanup)
- **Jul 28, 2025**: Last pre-rebuttal commit (wrong labels fix, big-LLM, memory bugs)
- **Feb 2, 2026**: Rebuttal commit for first submission round (scaling with k, label corruption, new PEFT variants, rebuttal plots)
- **Paper was REJECTED from the first venue.** The rebuttal materials in the repo (scatter_plot_rebuttal_part{1,2}.py, big-LLM bash_cmds, label corruption experiments, additional PEFT variants) are artifacts from that failed rebuttal.
- **Mar 2026**: Current — paper resubmitted to a new venue, now in peer review again with new reviewer requests for experiments.

## 12. bash_cmds Format: OLD (Legacy) vs NEW (makesbatch)

New rebuttal experiments will use the NEW makesbatch format from `/scratch/$USER/Test-Time-Sparsity/bash_cmds/`. Key differences:

### 12.1. OLD Format (Current in marc/bash_cmds/)

- **Directory structure**: `bash_cmds/{benchmark}/{N_experiment_type}/{hyperparams}.sh`
- **Job grouping**: `# comment` headers group clusters of commands; multiple commands per cluster allowed
- **MASTER_PORT**: Set manually in commands: `accelerate launch --main_process_port $MASTER_PORT ...`
- **Submission**: `python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/...`
- **Job names**: Free-form English descriptions (e.g., `# bbh gs8 lr1e-3 tokendrop0.1 seed42`)
- **Tracking**: `# [i/N] Submitted batch job JOBID -> ACCT`
- **2,346 files** across deeply nested directories

### 12.2. NEW Format (makesbatch from make_sbatch.py)

**File structure template:**
```bash
# Human-readable experiment description
# makesbatch --time 6 --ngpu 1 --gb 64 --bash_file bash_cmds/MMDD_N_experiment/file.sh

# unique_job_name_1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --benchmark bbh \
    --tag unique_job_name_1 \
    --model_name llama1b \
    --gs_epochs 16 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# unique_job_name_2
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --benchmark bbh \
    --tag unique_job_name_2 \
    --model_name llama1b \
    --gs_epochs 20 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

#! Submitted batch job 4985398 -> 36_mren -- unique_job_name_1
#! Submitted batch job 4985400 -> 36_general -- unique_job_name_2
```

**Rules:**
1. Header: descriptive comment + `# makesbatch ...` with exact submission command
2. Each job: `# job_name` on its own line, immediately followed by ONE command
3. Job names must be **globally unique** across ALL bash scripts
4. `--tag` value must match the job name exactly
5. Commands use `.venv/bin/accelerate launch` (full venv path, no bare `accelerate`)
6. NO `$MASTER_PORT` in commands (handled by makesbatch wrapper)
7. Blank lines between jobs for readability
8. `#!` tracking lines appended at bottom by makesbatch (never edit manually)
9. To skip a job: `# job_name` followed by `# SKIP: reason` (no command)
10. Selective resubmission: `makesbatch --only "pattern*" --bash_file ...`

**Directory naming**: `bash_cmds/MMDD_N_descriptive_name/` (date-prefixed, numbered)

### 12.3. Key Argument Patterns for marc Experiments

Arguments that are always needed per benchmark:

**BBH** (script: `inference_bbh/test_time_evaluate.py`):
- `--seed {42,43,44,45,46}` (5 seeds, one job per seed typically)
- No weight_dir, no select_tasks_path

**MMLU** (script: `inference_mmlu/test_time_evaluate.py`):
- `--seed {42,43,44,45,46}`
- No weight_dir

**NLP-LR** (script: `inference_nlp/test_time_evaluate.py`):
- No seed arg, no weight_dir (single evaluation over 21 task families x 5 internal seeds)

**ARC** (script: `inference_arc/test_time_evaluate.py`):
- `--select_tasks_path data/task_info_part{1,2,3,4,5}.csv`
- `--no_bos`
- `--weight_dir 0317_noprogram_base --weight_epoch 24`

**Common CT-KV args**: `--gs_epochs`, `--gs_lr`, `--gs_dropout train`, `--gs_token_dropout`, `--gs_batch_size`
**Common TTT args**: `--ttt_iters`, `--ttt_lr 1e-4`, `--ttt_permute_n 1000`, `--ttt_batch_size`
**Common TTT+CT-KV args**: `--ttt_weight_dir` (loads saved TTT checkpoint), then GS args
**Prefix tuning**: `--gs_ntokens 32`, `--gs_dropout none`
**Zero-shot**: `--zero_shot`
**Eval on demos**: `--eval_on_demonstrations`
**Label corruption**: `--wrong_label {0.0,0.25,0.5,0.75,1.0}`
**Big LLM**: `--model_name {deepseek14b,qwen14b,...}`, `--untrainable_nbit 4` (for 32B models), `--batch_size 4`

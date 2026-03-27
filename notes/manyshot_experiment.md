# Many-Shot BBH Experiment (2026-03-26)

## Motivation

Test whether CT-KV continues to improve over ICL when the number of in-context demonstrations scales well beyond the standard few-shot regime (10-shot). Inspired by "Many-Shot In-Context Learning" (Agarwal et al., NeurIPS 2024, arXiv 2404.11018), which showed that ICL performance keeps improving with more demonstrations on BBH using Gemini 1.5 Pro (1M context). We test this with Llama 3 8B (8K context) and ask: does CT-KV still help when ICL already has many demonstrations?

## Setup

### Model
- `meta-llama/Meta-Llama-3-8B-Instruct` (default `--model_name llama8b`)
- 8192 token context window
- `--max_seq_len 8192`

### Task Selection

Selected the **8 shortest BBH tasks** (by average tokens per example) to maximize the number of demonstrations that fit within 8K context:

| Task | avg tok/ex | 50-shot total | 100-shot total | Max demos in 8K |
|------|----------:|-------------:|---------------:|----------------:|
| boolean_expressions | 17 | 886 | ~1,750 | 250 (all) |
| sports_understanding | 24 | 1,253 | ~2,460 | 250 (all) |
| multistep_arithmetic_two | 34 | 1,756 | ~3,450 | 238 |
| hyperbaton | 37 | 1,880 | ~3,700 | 222 |
| object_counting | 39 | 1,970 | ~3,880 | 210 |
| dyck_languages | 50 | 2,561 | ~5,050 | 161 |
| web_of_lies | 52 | 2,675 | ~5,280 | 155 |
| navigate | 57 | 3,054 | ~5,990 | 140 |

All 8 tasks fit 100-shot comfortably. Each task has 250 examples total (except causal_judgement/penguins/snarks, which are excluded). With 50 demos, 200 remain for test. With 100 demos, 150 remain.

### Demonstration Selection

BBH demonstrations are selected at runtime (not pre-split like NLP-LR/MetaICL). Per task: `random.seed(seed)` → shuffle all examples → first N become demos, rest become test. Different seeds give different splits.

## Experiments

### 50-shot

**ICL baseline** (`bbh_icl.sh`): 3 seeds (42-44)
- `--num_demonstrations 50 --max_seq_len 8192`

**CT-KV sweep** (`bbh_ctkv.sh`): seed 42 only, 9 configs
- Sweep: `gs_epochs` {10, 20, 40} x `gs_token_dropout` {0.1, 0.2, 0.3}
- Fixed: `--gs_lr 1e-3 --gs_batch_size 16 --gs_dropout train`
- Rationale for epoch range: original 10-shot BBH best was 16 epochs with batch_size=2 (80 total steps). At 50-shot with batch_size=16, steps/epoch = ceil(50/16) = 4. So {10, 20, 40} epochs = {40, 80, 160} total steps, bracketing the original.
- Rationale for token dropout range: more demos = larger KV = more parameters to optimize, so 0.1 is the minimum (original best). Higher dropout may help regularize.
- VRAM: ~93 GB peak at batch_size=16 on H200 (verified by smoke test)

### 100-shot

**ICL baseline** (`bbh_icl_100shot.sh`): 3 seeds (42-44)
- `--num_demonstrations 100 --max_seq_len 8192`

**CT-KV sweep** (`bbh_ctkv_100shot.sh`): seed 42 only, 9 configs
- Same sweep grid as 50-shot
- `--gs_batch_size 8` (reduced from 16 due to ~2x larger KV cache, estimated ~114 GB at bs=8)
- Steps/epoch = ceil(100/8) = 13

## Code Changes

### `inference_bbh/data_utils.py`
- Added `select_tasks: Optional[List[str]]` parameter to `EvalDataset.__init__`
- When provided, only loads those tasks. Default `None` = all 27 tasks (backwards compatible)
- Stores active task list as `self.task_names`

### `inference_bbh/test_time_evaluate.py`
- Added `--select_tasks` argparse argument (`nargs='+'`, default `None`)
- Replaced `TASKS` / `TASKS.keys()` references in eval loop (lines 631, 640, 645) with `dataset.task_names`
- Fixed latent bug: `batch_past_key_values_attention_mask` in `run_gs()` was allocated with `batch_size` (the arg) instead of `bs` (actual batch size). Never triggered before because 10 demos / batch_size=2 = even batches. With 50 demos / batch_size=8, last batch has 2 items → assertion failure. Fixed on lines 1895, 2008-2009, 2031.
- Added in-process GPU keepalive in `__main__` block

## Files

```
bash_cmds/0326_1_bbh_manyshot/
  bbh_icl.sh           # 50-shot ICL, 3 seeds
  bbh_ctkv.sh          # 50-shot CT-KV sweep, 9 configs
  bbh_icl_100shot.sh   # 100-shot ICL, 3 seeds
  bbh_ctkv_100shot.sh  # 100-shot CT-KV sweep, 9 configs
```

## Next Steps

1. Once sweep results come in, pick best (epochs, token_dropout) for each shot count
2. Run best config with 3 seeds for statistical significance
3. Compare ICL vs CT-KV across 50-shot and 100-shot
4. Consider extending to more shot counts (e.g., 25, 150) to trace a full scaling curve

# Demonstration Ordering Sensitivity Study

**Date**: 2026-03-29
**Motivation**: Reviewer comment: "The method relies on demonstration contexts, yet the paper does not thoroughly analyze sensitivity to demonstration quality, ordering, or selection."
**Goal**: Show that ICL, CT-KV, and Prefix Tuning all have similar (low) variance across demo orderings, and that CT-KV's accuracy advantage is maintained regardless of ordering.

## 1. Code Changes

Added `--demo_shuffle_seed` argument to BBH and MMLU pipelines. When set, it re-shuffles the already-selected demos with a separate seed, keeping the **same set** of demos but changing their **order**. When `None` (default), behavior is unchanged (backwards compatible).

### Files modified

**`inference_bbh/data_utils.py`** — `EvalDataset.__init__`:
- Added `demo_shuffle_seed: Optional[int] = None` parameter
- After `random.seed(seed); random.shuffle(all_examples); demos = all_examples[:num_demonstrations]`, added:
  ```python
  if demo_shuffle_seed is not None:
      random.seed(demo_shuffle_seed)
      random.shuffle(self.task_to_demonstrations[task_name])
  ```

**`inference_bbh/test_time_evaluate.py`**:
- Added argparse: `--demo_shuffle_seed` (int, default=None)
- Passed `demo_shuffle_seed=args.demo_shuffle_seed` to `EvalDataset()`

**`inference_mmlu/data_utils.py`** — `EvalDataset.__init__`:
- Added `demo_shuffle_seed: Optional[int] = None` parameter
- Same re-shuffle logic after demo selection (line ~245)

**`inference_mmlu/test_time_evaluate.py`**:
- Added argparse: `--demo_shuffle_seed` (int, default=None)
- Passed `demo_shuffle_seed=args.demo_shuffle_seed` to `EvalDataset()`

### Why LOO is unaffected

LOO masking depends on `demon_start_idxs` (token positions of each demo in the KV cache) and `GSDataset.example_idx` (which demo each pair corresponds to). Both are derived from the demo list **after** the shuffle, so they remain consistent. Re-ordering demos before tokenization means the KV cache positions and LOO indices naturally reflect the new order.

## 2. Experiment Design

- **Benchmarks**: BBH (23 tasks, 10 demos each), MMLU (57 subjects, 16 demos each)
- **Methods**: ICL (no optimization), CT-KV (gradient-optimize KV cache), Prefix Tuning (optimize 32 learned tokens)
- **Demo selection seed**: `--seed 42` (fixed across all runs, same demos selected)
- **Demo ordering seeds**: `--demo_shuffle_seed {0, 1, 2, 3, 4}` (5 different permutations)
- **Model**: Llama-3.2-1B-Instruct (default for both benchmarks)

### Hyperparameters

**CT-KV** (from `bash_cmds/0327_1_adaptive_loo`):
- BBH: `--gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1`
- MMLU: `--gs_epochs 20 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1` (batch_size=default=8)

**Prefix Tuning** (same iters as CT-KV for fair comparison):
- BBH: `--gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_ntokens 32 --gs_dropout none --gs_token_dropout 0.1`
- MMLU: `--gs_epochs 20 --gs_lr 1e-3 --gs_ntokens 32 --gs_dropout none --gs_token_dropout 0.1`

**ICL**: No optimization args (just forward pass with demos as context).

## 3. Bash Files

All in `bash_cmds/0328_0_demo_ordering/`. Each file runs 5 orderings (dss0-4) sequentially via `--single`.

| File | Method | Benchmark | Submit Command |
|------|--------|-----------|---------------|
| `bbh_icl.sh` | ICL | BBH | `makesbatch --time 6 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0328_0_demo_ordering/bbh_icl.sh` |
| `bbh_ctkv.sh` | CT-KV | BBH | `makesbatch --time 6 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0328_0_demo_ordering/bbh_ctkv.sh` |
| `bbh_prefix.sh` | Prefix | BBH | `makesbatch --time 6 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0328_0_demo_ordering/bbh_prefix.sh` |
| `mmlu_icl.sh` | ICL | MMLU | `makesbatch --time 6 --ngpu 1 --gb 64 --single --bash_file bash_cmds/0328_0_demo_ordering/mmlu_icl.sh` |
| `mmlu_ctkv.sh` | CT-KV | MMLU | `makesbatch --time 6 --ngpu 1 --gb 64 --single --bash_file bash_cmds/0328_0_demo_ordering/mmlu_ctkv.sh` |
| `mmlu_prefix.sh` | Prefix | MMLU | `makesbatch --time 6 --ngpu 1 --gb 64 --single --bash_file bash_cmds/0328_0_demo_ordering/mmlu_prefix.sh` |

## 4. Job Execution

| File | Job ID | Account | Partition | Node | Elapsed | State |
|------|--------|---------|-----------|------|---------|-------|
| bbh_icl.sh | 5072887 | 36_mren | l40s_courant | gl055 | 1:11:21 | COMPLETED 0:0 |
| bbh_ctkv.sh | 5072895 | 36_mren | l40s_courant | gl009 | 1:20:27 | COMPLETED 0:0 |
| bbh_prefix.sh | 5072903 | 36_mren | l40s_courant | gl051 | 1:19:09 | COMPLETED 0:0 |
| mmlu_icl.sh | 5072912 | 36_cds | h200_cds | gh120 | 0:40:28 | COMPLETED 0:0 |
| mmlu_ctkv.sh | 5072921 | 36_cds | h200_cds | gh124 | 0:57:19 | COMPLETED 0:0 |
| mmlu_prefix.sh | 5072931 | 36_cds | h200_cds | gh105 | 0:53:13 | COMPLETED 0:0 |

All 30 runs completed. SLURM logs: `slurm_outs/{5072887,5072895,5072903,5072912,5072921,5072931}.out`.
Output directories: `encoder_decoder/outputs_eval/eval_{bbh,mmlu}_order_{icl,ctkv,prefix}_dss{0..4}/`.

## 5. Raw Results

### BBH (eval/score from slurm logs, order = dss0, dss1, dss2, dss3, dss4)

| Method | dss0 | dss1 | dss2 | dss3 | dss4 |
|--------|------|------|------|------|------|
| ICL | 49.2785 | 49.3727 | 49.7002 | 49.2307 | 48.4063 |
| CT-KV | 56.7339 | 55.5319 | 55.7271 | 56.2272 | 56.2164 |
| Prefix | 49.6588 | 49.2859 | 49.7696 | 49.2423 | 48.5717 |

Source: `slurm_outs/5072887.out` (ICL), `slurm_outs/5072895.out` (CT-KV), `slurm_outs/5072903.out` (Prefix)

### MMLU (eval/score from slurm logs, order = dss0, dss1, dss2, dss3, dss4)

| Method | dss0 | dss1 | dss2 | dss3 | dss4 |
|--------|------|------|------|------|------|
| ICL | 40.7878 | 40.4718 | 40.2811 | 40.2431 | 41.0059 |
| CT-KV | 43.4895 | 43.2275 | 42.5277 | 42.6483 | 42.8378 |
| Prefix | 41.1446 | 40.3504 | 40.4091 | 40.2814 | 40.9673 |

Source: `slurm_outs/5072912.out` (ICL), `slurm_outs/5072921.out` (CT-KV), `slurm_outs/5072931.out` (Prefix)

## 6. Summary Tables

### BBH

| Method | Mean | Std |
|--------|------|-----|
| ICL | 49.20 | 0.47 |
| CT-KV | 56.09 | 0.46 |
| Prefix | 49.31 | 0.46 |

### MMLU

| Method | Mean | Std |
|--------|------|-----|
| ICL | 40.56 | 0.32 |
| CT-KV | 42.95 | 0.39 |
| Prefix | 40.63 | 0.38 |

## 7. Interpretation

- All three methods show very low variance across orderings (std 0.3-0.5 for all).
- ICL, CT-KV, and Prefix Tuning are all similarly robust to demo ordering on these benchmarks with 10 (BBH) / 16 (MMLU) demos.
- CT-KV maintains its accuracy advantage regardless of ordering: +6.9pp over ICL on BBH, +2.4pp on MMLU.
- Prefix Tuning barely improves over ICL on either benchmark.
- The low ICL variance is likely because 10-16 demos provide enough context that individual ordering effects average out. Sensitivity would likely be more pronounced with fewer demos (2-4).

## 8. Rebuttal Argument

The experiment shows that all methods — ICL, CT-KV, and Prefix Tuning — exhibit comparable and low sensitivity to demonstration ordering (std < 0.5% across 5 random permutations on both BBH and MMLU). CT-KV's accuracy gains over baselines (+6.9% on BBH, +2.4% on MMLU) are consistent across all orderings, demonstrating that the method's improvements are not artifacts of a particular demonstration arrangement. This complements the existing label corruption robustness study (Section 5.6/6.2), which shows CT-KV degrades gracefully under noisy demonstrations.

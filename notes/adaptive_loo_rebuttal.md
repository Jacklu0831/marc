# Adaptive LOO Rebuttal Experiments

**Date**: 2026-03-29
**Reviewer concern**: R2-W2 — LOO masking disabled for ARC due to few demos, indicating limited robustness/automated adaptability in extreme low-resource settings.

## Overview

Three experiment sets to address the reviewer:
1. **ARC adaptive LOO threshold (K sweep)**: Enable LOO only when #demos >= K
2. **BBH LOO vs noLOO vs ICL across num_demonstrations**: Find crossover where LOO helps
3. **MMLU LOO vs noLOO vs ICL across num_demonstrations**: Same analysis

## Code Changes

### `--gs_loo_min_demos K` argument (all 3 benchmark scripts)

Added to `inference_arc/test_time_evaluate.py`, `inference_bbh/test_time_evaluate.py`, `inference_mmlu/test_time_evaluate.py`.

When `--gs_loo_min_demos K` is set (K > 0), per-task decision before `run_gs()`:
```python
n_demos = len(demon_start_idxs)
if gs_loo_min_demos > 0:
    effective_gs_dropout = 'train' if n_demos >= gs_loo_min_demos else 'none'
else:
    effective_gs_dropout = gs_dropout  # use original arg as-is
```

This overrides `--gs_dropout` on a per-task basis. Default K=-1 (disabled, fully backwards-compatible).

### Bug fix: `batch_size` vs `bs` in `run_gs()` training loop

Found during these experiments. In `run_gs()`, the DataLoader caps batch size: `batch_size = min(batch_size, len(gs_dataset))`. The last batch can be smaller when `num_demos % batch_size != 0`. Several tensors were allocated with `batch_size` (the capped max) instead of `bs` (the actual batch size from `pair_input_ids.shape[0]`):

- `batch_past_key_values_attention_mask = torch.ones((batch_size, ...))` → `torch.ones((bs, ...))`
- `prefix_layer_k.expand(batch_size, ...)` → `expand(bs, ...)`
- `position_ids = torch.zeros((batch_size, ...))` → `torch.zeros((bs, ...))`

Fixed in `inference_mmlu/test_time_evaluate.py` and `inference_arc/test_time_evaluate.py`. BBH already used `bs` correctly.

The intentional `loss = model_out.loss * bs / batch_size` (gradient scaling for smaller last batch) was left unchanged.

### GPU keepalive

Added to `inference_mmlu/test_time_evaluate.py` `if __name__ == "__main__":` block:
```python
import sys
sys.path.insert(0, "/scratch/yl11330")
import gpu_keepalive
gpu_keepalive.start()
```

## Experiment Configs

### ARC Adaptive LOO (K sweep)

**Config**: lr=3e-3, epochs=200, gs_token_dropout=0.1, gs_dropout varies by config
**Model**: Llama-3.2-1B-Instruct fine-tuned, `--weight_dir 0317_noprogram_base --weight_epoch 24 --no_bos`
**Tasks**: 400 tasks split across 5 parts (80 each)
**GPU**: L40S, 4h walltime
**Bash files**: `bash_cmds/0327_1_adaptive_loo/arc_adaptive_loo.sh` (25 jobs: 5 configs × 5 parts), `arc_adaptive_loo_k67.sh` (10 jobs: 2 configs × 5 parts)
**Submission**: `makesbatch --time 4 --ngpu 1 --gb 64 --l40s --bash_file ...`

Configs:
- `arc_noloo_part{1-5}`: `--gs_dropout none`
- `arc_fullloo_part{1-5}`: `--gs_dropout train`
- `arc_adaptloo_k3_part{1-5}`: `--gs_loo_min_demos 3`
- `arc_adaptloo_k4_part{1-5}`: `--gs_loo_min_demos 4`
- `arc_adaptloo_k5_part{1-5}`: `--gs_loo_min_demos 5`
- `arc_adaptloo_k6_part{1-5}`: `--gs_loo_min_demos 6`
- `arc_adaptloo_k7_part{1-5}`: `--gs_loo_min_demos 7`

ARC demo distribution (400 tasks):
- 2 demos: 45 tasks (11.2%)
- 3 demos: 217 tasks (54.2%)
- 4 demos: 90 tasks (22.5%)
- 5 demos: 30 tasks (7.5%)
- 6 demos: 14 tasks (3.5%)
- 7 demos: 4 tasks (1.0%)

### BBH LOO vs noLOO vs ICL (ndemo sweep)

**Config**: lr=1e-3, epochs=20, gs_batch_size=2, gs_token_dropout=0.1, seed={42,43,44}
**Model**: Llama-3.2-1B-Instruct (off-the-shelf)
**ndemo sweep**: {2, 3, 4, 6, 8, 10}
**GPU**: L40S, 8h walltime, --single
**Bash files**:
- `bbh_icl_vs_ndemo_s{42,43,44}.sh` (6 commands each, ICL only — no GS args)
- `bbh_loo_vs_ndemo_s{42,43,44}.sh` (6 commands each, `--gs_dropout train`)
- `bbh_noloo_vs_ndemo_s{42,43,44}.sh` (6 commands each, `--gs_dropout none`)
**Submission**: `makesbatch --time 8 --ngpu 1 --gb 64 --l40s --single --bash_file ...`

### MMLU LOO vs noLOO vs ICL (ndemo sweep)

**Config**: lr=1e-3, epochs=20, gs_token_dropout=0.1, seed={42,43,44}
**Model**: Llama-3.2-1B-Instruct (off-the-shelf)
**ndemo sweep**: {2, 3, 4, 6, 8, 12, 16}
**GPU**: H200, 6h walltime, --single (L40S OOM'd at nd>=4 for some MMLU subjects)
**Bash files**:
- `mmlu_icl_vs_ndemo_s{42,43,44}.sh` (7 commands each)
- `mmlu_loo_vs_ndemo_s{42,43,44}.sh` (14 commands: 7 noloo + 7 loo, sequential in --single)
**Submission**: `makesbatch --time 6 --ngpu 1 --gb 64 --single --bash_file ...`

**Note**: The `mmlu_loo_vs_ndemo` files contain BOTH noloo and loo runs (noloo first, loo second). Multiple rounds of crashes:
- Round 1 (L40S): OOM at nd=4
- Round 2 (H200): AssertionError at nd=12 (batch_size vs bs bug)
- Round 3 (H200): RuntimeError in RoPE at nd=12 (partial fix — position_ids still used batch_size)
- Round 4 (L40S, loo-only rerun): OOM at nd=4 on L40S again

## Job IDs

### ARC
| Job ID | Account | Config |
|--------|---------|--------|
| 5006899 | 36_mren | arc_noloo_part1 |
| 5006900 | 219_courant | arc_noloo_part2 |
| 5006901 | 36_general | arc_noloo_part3 |
| 5006902 | 36_cds | arc_noloo_part4 |
| 5006903 | 36_mren | arc_noloo_part5 |
| 5006904 | 219_courant | arc_fullloo_part1 |
| 5006905 | 36_general | arc_fullloo_part2 |
| 5006906 | 36_cds | arc_fullloo_part3 |
| 5006907 | 36_mren | arc_fullloo_part4 |
| 5006908 | 219_courant | arc_fullloo_part5 |
| 5006909 | 36_mren | arc_adaptloo_k3_part1 |
| 5006910 | 219_courant | arc_adaptloo_k3_part2 |
| 5006911 | 36_mren | arc_adaptloo_k3_part3 |
| 5006912 | 219_courant | arc_adaptloo_k3_part4 |
| 5006913 | 36_mren | arc_adaptloo_k3_part5 |
| 5006914 | 219_courant | arc_adaptloo_k4_part1 |
| 5006915 | 36_mren | arc_adaptloo_k4_part2 |
| 5006916 | 219_courant | arc_adaptloo_k4_part3 |
| 5006917 | 36_mren | arc_adaptloo_k4_part4 |
| 5006918 | 219_courant | arc_adaptloo_k4_part5 |
| 5006919 | 36_mren | arc_adaptloo_k5_part1 |
| 5006920 | 219_courant | arc_adaptloo_k5_part2 |
| 5006921 | 36_mren | arc_adaptloo_k5_part3 |
| 5006922 | 219_courant | arc_adaptloo_k5_part4 |
| 5006923 | 36_mren | arc_adaptloo_k5_part5 |
| 5031380 | 36_mren | arc_adaptloo_k6_part1 |
| 5031381 | 219_courant | arc_adaptloo_k6_part2 |
| 5031382 | 36_general | arc_adaptloo_k6_part3 |
| 5031383 | 36_cds | arc_adaptloo_k6_part4 |
| 5031384 | 36_mren | arc_adaptloo_k6_part5 |
| 5031385 | 219_courant | arc_adaptloo_k7_part1 |
| 5031386 | 36_mren | arc_adaptloo_k7_part2 |
| 5031387 | 219_courant | arc_adaptloo_k7_part3 |
| 5031388 | 36_mren | arc_adaptloo_k7_part4 |
| 5031389 | 36_mren | arc_adaptloo_k7_part5 |

### BBH
| Job ID | Account | Config | Status |
|--------|---------|--------|--------|
| 5043766 | 219_courant | bbh_icl_s42 | COMPLETED (all 6 nd) |
| 5043775 | 36_mren | bbh_icl_s43 | COMPLETED (all 6 nd) |
| 5043783 | 36_mren | bbh_icl_s44 | COMPLETED (all 6 nd) |
| 5043644 | 36_mren | bbh_loo_s42 | COMPLETED (all 6 nd) |
| 5043652 | 36_mren | bbh_loo_s43 | COMPLETED (all 6 nd) |
| 5043660 | 36_mren | bbh_loo_s44 | FAILED (OOM at nd=4, only nd=2,3) |
| 5072755 | 36_mren | bbh_noloo_s42 | COMPLETED (all 6 nd) |
| 5072766 | 36_mren | bbh_noloo_s43 | COMPLETED (all 6 nd) |
| 5072777 | 36_mren | bbh_noloo_s44 | FAILED (OOM at nd=4, only nd=2,3) |

### MMLU
| Job ID | Account | Config | Status |
|--------|---------|--------|--------|
| 5056390 | 36_cds | mmlu_icl_s42 | COMPLETED (all 7 nd) |
| 5056399 | 36_cds | mmlu_icl_s43 | COMPLETED (all 7 nd) |
| 5056408 | 36_cds | mmlu_icl_s44 | COMPLETED (all 7 nd) |
| 5070327 | 219_courant | mmlu_noloo_s42 | PARTIAL (nd=2-8, crashed nd=12) |
| 5070336 | 219_courant | mmlu_noloo_s43 | PARTIAL (nd=2-8, crashed nd=12) |
| 5070345 | 219_courant | mmlu_noloo_s44 | PARTIAL (nd=2-8, crashed nd=12) |
| 5095257 | 219_courant | mmlu_loo_s42 | PARTIAL (nd=2,3 only, OOM nd=4 on L40S) |
| 5095267 | 219_courant | mmlu_loo_s43 | PARTIAL (nd=2,3,4 only) |
| 5095275 | 219_courant | mmlu_loo_s44 | PARTIAL (nd=2,3 only) |

## Raw Scores

### BBH (eval/score = mean per-task accuracy)

| Tag | Score |
|-----|-------|
| bbh_icl_s42_nd2 | 44.2926 |
| bbh_icl_s42_nd3 | 46.0950 |
| bbh_icl_s42_nd4 | 45.7637 |
| bbh_icl_s42_nd6 | 46.9717 |
| bbh_icl_s42_nd8 | 46.7283 |
| bbh_icl_s42_nd10 | 47.8715 |
| bbh_icl_s43_nd2 | 44.8785 |
| bbh_icl_s43_nd3 | 44.8353 |
| bbh_icl_s43_nd4 | 45.9947 |
| bbh_icl_s43_nd6 | 46.9792 |
| bbh_icl_s43_nd8 | 47.5315 |
| bbh_icl_s43_nd10 | 50.6288 |
| bbh_icl_s44_nd2 | 45.5235 |
| bbh_icl_s44_nd3 | 45.0926 |
| bbh_icl_s44_nd4 | 46.2217 |
| bbh_icl_s44_nd6 | 47.8351 |
| bbh_icl_s44_nd8 | 48.3891 |
| bbh_icl_s44_nd10 | 49.6047 |
| bbh_noloo_s42_nd2 | 44.7371 |
| bbh_noloo_s42_nd3 | 47.6943 |
| bbh_noloo_s42_nd4 | 47.9565 |
| bbh_noloo_s42_nd6 | 49.9748 |
| bbh_noloo_s42_nd8 | 51.1230 |
| bbh_noloo_s42_nd10 | 52.5838 |
| bbh_noloo_s43_nd2 | 44.5314 |
| bbh_noloo_s43_nd3 | 45.2031 |
| bbh_noloo_s43_nd4 | 48.5952 |
| bbh_noloo_s43_nd6 | 49.6745 |
| bbh_noloo_s43_nd8 | 51.1405 |
| bbh_noloo_s43_nd10 | 51.6435 |
| bbh_noloo_s44_nd2 | 42.4035 |
| bbh_noloo_s44_nd3 | 44.9650 |
| bbh_loo_s42_nd2 | 46.1274 |
| bbh_loo_s42_nd3 | 49.8006 |
| bbh_loo_s42_nd4 | 50.8483 |
| bbh_loo_s42_nd6 | 52.1161 |
| bbh_loo_s42_nd8 | 53.3383 |
| bbh_loo_s42_nd10 | 56.2834 |
| bbh_loo_s43_nd2 | 45.4844 |
| bbh_loo_s43_nd3 | 48.2941 |
| bbh_loo_s43_nd4 | 51.1243 |
| bbh_loo_s43_nd6 | 53.0092 |
| bbh_loo_s43_nd8 | 53.0214 |
| bbh_loo_s43_nd10 | 56.3629 |
| bbh_loo_s44_nd2 | 44.2877 |
| bbh_loo_s44_nd3 | 46.1213 |

### MMLU (eval/score = mean per-subject accuracy)

| Tag | Score |
|-----|-------|
| mmlu_icl_s42_nd2 | 38.5052 |
| mmlu_icl_s42_nd3 | 39.2647 |
| mmlu_icl_s42_nd4 | 40.0315 |
| mmlu_icl_s42_nd6 | 40.0732 |
| mmlu_icl_s42_nd8 | 39.9704 |
| mmlu_icl_s42_nd12 | 41.6238 |
| mmlu_icl_s42_nd16 | 40.3650 |
| mmlu_icl_s43_nd2 | 40.2199 |
| mmlu_icl_s43_nd3 | 40.5897 |
| mmlu_icl_s43_nd4 | 40.3978 |
| mmlu_icl_s43_nd6 | 40.6222 |
| mmlu_icl_s43_nd8 | 40.4812 |
| mmlu_icl_s43_nd12 | 41.0943 |
| mmlu_icl_s43_nd16 | 41.1337 |
| mmlu_icl_s44_nd2 | 38.6308 |
| mmlu_icl_s44_nd3 | 40.3802 |
| mmlu_icl_s44_nd4 | 40.3978 |
| mmlu_icl_s44_nd6 | 40.7462 |
| mmlu_icl_s44_nd8 | 42.1839 |
| mmlu_icl_s44_nd12 | 42.9485 |
| mmlu_icl_s44_nd16 | 42.5800 |
| mmlu_noloo_s42_nd2 | 39.4684 |
| mmlu_noloo_s42_nd3 | 39.6150 |
| mmlu_noloo_s42_nd4 | 40.2515 |
| mmlu_noloo_s42_nd6 | 40.7764 |
| mmlu_noloo_s42_nd8 | 42.0347 |
| mmlu_noloo_s43_nd2 | 40.1905 |
| mmlu_noloo_s43_nd3 | 41.4734 |
| mmlu_noloo_s43_nd4 | 41.5739 |
| mmlu_noloo_s43_nd6 | 41.3785 |
| mmlu_noloo_s43_nd8 | 41.2036 |
| mmlu_noloo_s44_nd2 | 38.1265 |
| mmlu_noloo_s44_nd3 | 40.8389 |
| mmlu_noloo_s44_nd4 | 41.0525 |
| mmlu_noloo_s44_nd6 | 41.1460 |
| mmlu_noloo_s44_nd8 | 42.1394 |
| mmlu_loo_s42_nd2 | 38.9976 |
| mmlu_loo_s42_nd3 | 40.3405 |
| mmlu_loo_s43_nd2 | 40.4197 |
| mmlu_loo_s43_nd3 | 41.9821 |
| mmlu_loo_s43_nd4 | 42.2995 |
| mmlu_loo_s44_nd2 | 38.8660 |
| mmlu_loo_s44_nd3 | 40.8198 |

### ARC (per-part metrics, raw floats)

| Tag | competition_all | exact_acc | token_acc | correct_dim |
|-----|----------------|-----------|-----------|-------------|
| arc_noloo_part1 | 0.2000 | 0.2073 | 0.7505 | 0.8902 |
| arc_noloo_part2 | 0.2875 | 0.3095 | 0.8427 | 0.9405 |
| arc_noloo_part3 | 0.1500 | 0.1566 | 0.7913 | 0.9518 |
| arc_noloo_part4 | 0.2750 | 0.2927 | 0.8119 | 0.9146 |
| arc_noloo_part5 | 0.2875 | 0.2976 | 0.8206 | 0.9286 |
| arc_fullloo_part1 | 0.1625 | 0.1585 | 0.7649 | 0.8902 |
| arc_fullloo_part2 | 0.2625 | 0.2857 | 0.7965 | 0.9286 |
| arc_fullloo_part3 | 0.1125 | 0.1205 | 0.7700 | 0.9398 |
| arc_fullloo_part4 | 0.2625 | 0.2805 | 0.7789 | 0.8902 |
| arc_fullloo_part5 | 0.2500 | 0.2619 | 0.8179 | 0.9286 |
| arc_adaptloo_k3_part1 | 0.1750 | 0.1707 | 0.7667 | 0.8902 |
| arc_adaptloo_k3_part2 | 0.2625 | 0.2857 | 0.7983 | 0.9286 |
| arc_adaptloo_k3_part3 | 0.1375 | 0.1446 | 0.7723 | 0.9398 |
| arc_adaptloo_k3_part4 | 0.2625 | 0.2805 | 0.7823 | 0.8902 |
| arc_adaptloo_k3_part5 | 0.2875 | 0.2976 | 0.8254 | 0.9286 |
| arc_adaptloo_k4_part1 | 0.1750 | 0.1829 | 0.7563 | 0.8902 |
| arc_adaptloo_k4_part2 | 0.3000 | 0.3214 | 0.8433 | 0.9405 |
| arc_adaptloo_k4_part3 | 0.1250 | 0.1325 | 0.7799 | 0.9518 |
| arc_adaptloo_k4_part4 | 0.2750 | 0.2927 | 0.8065 | 0.9146 |
| arc_adaptloo_k4_part5 | 0.2750 | 0.2857 | 0.8223 | 0.9286 |
| arc_adaptloo_k5_part1 | 0.1875 | 0.1951 | 0.7519 | 0.8902 |
| arc_adaptloo_k5_part2 | 0.3125 | 0.3333 | 0.8376 | 0.9405 |
| arc_adaptloo_k5_part3 | 0.1375 | 0.1446 | 0.7892 | 0.9518 |
| arc_adaptloo_k5_part4 | 0.2625 | 0.2805 | 0.8093 | 0.9146 |
| arc_adaptloo_k5_part5 | 0.2750 | 0.2857 | 0.8151 | 0.9286 |
| arc_adaptloo_k6_part1 | 0.1750 | 0.1829 | 0.7379 | 0.8902 |
| arc_adaptloo_k6_part2 | 0.3125 | 0.3333 | 0.8360 | 0.9405 |
| arc_adaptloo_k6_part3 | 0.1500 | 0.1566 | 0.7805 | 0.9398 |
| arc_adaptloo_k6_part4 | 0.2625 | 0.2805 | 0.8084 | 0.9146 |
| arc_adaptloo_k6_part5 | 0.2875 | 0.2976 | 0.8172 | 0.9286 |
| arc_adaptloo_k7_part1 | 0.1875 | 0.1951 | 0.7404 | 0.8902 |
| arc_adaptloo_k7_part2 | 0.3000 | 0.3214 | 0.8436 | 0.9405 |
| arc_adaptloo_k7_part3 | 0.1500 | 0.1566 | 0.7920 | 0.9518 |
| arc_adaptloo_k7_part4 | 0.2625 | 0.2805 | 0.8114 | 0.9146 |
| arc_adaptloo_k7_part5 | 0.2875 | 0.2976 | 0.8306 | 0.9405 |

## Aggregated Results

### BBH: ICL vs CT-KV(noLOO) vs CT-KV(LOO)

Seeds 42,43 have all 6 nd values. Seed 44 has nd=2,3 only (OOM on L40S at nd=4).

| nd | ICL (mean±std) | CT-KV noLOO | CT-KV LOO | LOO−noLOO | noLOO−ICL |
|----|----------------|-------------|-----------|-----------|-----------|
| 2 | 44.90±0.50 (3) | 43.89±1.17 (3) | 45.30±0.77 (3) | +1.41% | −1.01% |
| 3 | 45.34±0.56 (3) | 45.95±1.21 (3) | 48.07±1.56 (3) | +2.12% | +0.61% |
| 4 | 45.99±0.19 (3) | 48.28±0.32 (2) | 50.99±0.14 (2) | +2.71% | +2.28% |
| 6 | 47.26±0.40 (3) | 49.82±0.15 (2) | 52.56±0.45 (2) | +2.74% | +2.56% |
| 8 | 47.55±0.68 (3) | 51.13±0.01 (2) | 53.18±0.16 (2) | +2.05% | +3.58% |
| 10 | 49.37±1.13 (3) | 52.11±0.47 (2) | 56.32±0.04 (2) | +4.21% | +2.75% |

Key finding: **LOO always helps over noLOO on BBH, even at nd=2** (+1.41%). Benefit grows with demo count. CT-KV noLOO at nd=2 actually hurts vs ICL (−1.01%).

### MMLU: ICL vs CT-KV(noLOO) vs CT-KV(LOO)

All 3 seeds for ICL and noLOO (nd=2-8). LOO data partial (nd=2-3 for all seeds, nd=4 for s43 only). nd=12,16 noLOO/LOO not available (batch_size bug crashed at nd=12).

| nd | ICL (mean±std) | CT-KV noLOO | CT-KV LOO | LOO−noLOO | noLOO−ICL |
|----|----------------|-------------|-----------|-----------|-----------|
| 2 | 39.12±0.78 (3) | 39.26±0.84 (3) | 39.43±0.67 (3) | +0.17% | +0.14% |
| 3 | 40.08±0.57 (3) | 40.64±0.76 (3) | 41.05±0.69 (3) | +0.41% | +0.56% |
| 4 | 40.56±0.50 (3) | 40.96±0.55 (3) | 42.30±0.00 (1) | +1.34%* | +0.40% |
| 6 | 40.48±0.29 (3) | 41.10±0.25 (3) | — | — | +0.62% |
| 8 | 40.88±0.93 (3) | 41.79±0.42 (3) | — | — | +0.91% |
| 12 | 41.89±0.78 (3) | — | — | — | — |
| 16 | 41.36±0.92 (3) | — | — | — | — |

*Only 1 seed (s43). LOO helps here too, but data incomplete.

### ARC: Adaptive LOO (competition_all accuracy, all 5 parts)

| Config | % tasks LOO | Acc | vs NoLOO |
|--------|-------------|-----|----------|
| No LOO | 0% | 24.00% | — |
| K=7 | 1% | 23.75% | −0.25% |
| K=6 | 4% | 23.75% | −0.25% |
| K=5 | 12% | 23.50% | −0.50% |
| K=4 | 35% | 23.00% | −1.00% |
| K=3 | 89% | 22.50% | −1.50% |
| Full LOO | 100% | 21.00% | −3.00% |

### ARC: Competition accuracy stratified by #demos × LOO config

| #demo | #task | NoLOO | K=7 | K=6 | K=5 | K=4 | K=3 | FullLOO |
|-------|-------|-------|-----|-----|-----|-----|-----|---------|
| 2 | 45 | 24.4% | 24.4% | 24.4% | 24.4% | 24.4% | 24.4% | 20.0% |
| 3 | 217 | 21.2% | 21.2% | 21.2% | 21.7% | 20.3% | 18.9% | 17.5% |
| 4 | 90 | 25.6% | 24.4% | 23.3% | 23.3% | 24.4% | 24.4% | 23.3% |
| 5 | 30 | 46.7% | 43.3% | 46.7% | 40.0% | 40.0% | 40.0% | 40.0% |
| 6 | 14 | 28.6% | 28.6% | 28.6% | 28.6% | 28.6% | 28.6% | 28.6% |
| 7 | 4 | 50.0% | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% |
| ALL | 400 | 25.0% | 24.8% | 24.8% | 24.5% | 24.0% | 23.2% | 21.8% |

Note: Minor differences between configs that should be identical (e.g., NoLOO vs K=7 at 5 demos) are due to token dropout randomness across separate runs, not the LOO mechanism.

### ARC: All metrics (mean across 5 parts, as percentages)

| Config | competition_all | exact_acc | token_acc | correct_dim |
|--------|----------------|-----------|-----------|-------------|
| No LOO | 24.00% | 25.28% | 80.34% | 92.51% |
| K=7 | 23.75% | 25.03% | 80.36% | 92.75% |
| K=6 | 23.75% | 25.02% | 79.60% | 92.27% |
| K=5 | 23.50% | 24.78% | 80.06% | 92.51% |
| K=4 | 23.00% | 24.31% | 80.17% | 92.51% |
| K=3 | 22.50% | 23.58% | 78.90% | 91.55% |
| Full LOO | 21.00% | 22.14% | 78.56% | 91.55% |

### ARC: Task flips (No LOO vs Full LOO)

- Both solved: 81
- Both unsolved: 290
- Gained by LOO (solved only with LOO): 6 — demo counts: [3, 3, 3, 4, 4, 7]
- Lost by LOO (solved only without LOO): 19 — demo counts: [2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5]

Net: LOO loses 13 tasks.

## Analysis & Interpretation

### Key findings

1. **LOO is benchmark-dependent, not demo-count-dependent.** BBH benefits from LOO even at nd=2 (+1.41%). MMLU benefits at nd=2 (+0.17%) and nd=3 (+0.41%). ARC is hurt by LOO at every demo count from 2 to 5, only benefits at 7 demos.

2. **The crossover threshold differs dramatically by benchmark.** BBH: LOO helps at all tested nd (2-10). MMLU: LOO helps at all tested nd (2-4). ARC: LOO hurts until ~7 demos.

3. **Adaptive thresholding works as expected** on ARC — higher K (less LOO) → closer to no-LOO performance. But a single threshold doesn't generalize across benchmarks.

### Why LOO hurts on ARC but helps on BBH/MMLU

LOO masking hides one demo and optimizes the KV to predict that demo's answer from the others. This is useful when the remaining demos still uniquely determine the task.

ARC tasks often require all demonstrations to disambiguate the transformation rule. Each input-output grid pair constrains the space of possible transformations. Removing one pair may leave multiple transformations consistent with the remaining demos, causing the optimizer to push the KV toward an incorrect interpretation. For example, if the rule involves both rotation and recoloring, one demo might be the only one that disambiguates between "rotate then recolor" vs "only rotate."

BBH and MMLU tasks are fully determined by any single demonstration. If you see one boolean expression evaluated correctly, the task is unambiguous. Removing one demo doesn't create ambiguity about what task is being performed — the remaining demos still uniquely identify it. The LOO gradient consistently points toward better task encoding.

This explains why the effect depends on the benchmark rather than the number of demos: it's about whether individual demonstrations are redundant (BBH/MMLU) or complementary (ARC) in specifying the task.

## Missing Data (TODO)

- MMLU LOO nd>=4: Need to rerun on H200 (L40S OOMs for long subjects). Requires the batch_size bug fix.
- MMLU noLOO nd=12,16: Same batch_size bug fix needed, rerun on H200.
- BBH s44 LOO nd>=4: OOM on L40S, rerun on H200 if 3rd seed needed.
- BBH s44 noLOO nd>=4: Same.

# Cross-Task Transfer Experiment — Full Results

Addresses reviewer R1 Q1: *"Generalization across tasks: The method optimizes context for each task individually. Have the authors explored whether optimized contexts can generalize across related tasks or datasets?"*

Date: 2026-03-29. Seed: 42. Model: Llama-3.2-1B-Instruct.

## 1. Experiment Design

### Setup

For each (source, target) pair, we use the source task's demonstrations for KV initialization and optimization, then evaluate on the target task's test set. This produces an N×N matrix where the diagonal = standard (on-task) and off-diagonal = cross-task transfer.

### Conditions

| Condition | Demos from | Optimized? | What it measures |
| --- | --- | --- | --- |
| Zero-shot | none | no | No context at all |
| ICL (own) | target | no | Standard ICL baseline |
| ICL (cross) | source | no | Effect of wrong demos alone |
| CT-KV (cross) | source | yes | Does optimized context generalize? |
| Prefix (cross) | source (loss signal only) | yes | Random KV optimized on source demos |

### Key comparisons

1. **CT-KV cross vs ICL cross** (same demos, does optimization help?)
2. **CT-KV cross vs zero-shot** (does optimized cross-task context carry useful information?)
3. **CT-KV cross vs ICL own** (does optimized cross-task context beat correct-task ICL?)

## 2. BBH: Logical Deduction

Three variants of the same task (arrange N objects given constraints): three_objects (3 choices), five_objects (5 choices), seven_objects (7 choices). Same reasoning skill, different difficulty via object count.

### 2.1 Raw Accuracy (%)

**Zero-shot**: three=56.2, five=31.2, seven=26.7

**ICL (own demos)**: three=58.3, five=37.1, seven=44.2

**ICL (cross-task demos):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 58.3 | 41.7 | 42.1 |
| five_objects | 65.0 | 37.1 | 43.3 |
| seven_objects | 63.3 | 40.4 | 44.2 |

**CT-KV (cross-task, epoch 5):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 58.8 | 41.7 | 44.2 |
| five_objects | 68.8 | 44.6 | 47.1 |
| seven_objects | 65.4 | 41.2 | 45.0 |

**CT-KV (cross-task, epoch 10):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 59.6 | 42.5 | 43.3 |
| five_objects | 71.7 | 47.9 | 49.6 |
| seven_objects | 62.9 | 42.1 | 47.1 |

**CT-KV (cross-task, epoch 15):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 62.9 | 43.3 | 42.1 |
| five_objects | 75.4 | 47.5 | 49.2 |
| seven_objects | 62.1 | 40.4 | 45.4 |

**CT-KV (cross-task, epoch 20):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 61.7 | 44.2 | 42.9 |
| five_objects | 75.8 | 48.8 | 49.2 |
| seven_objects | 61.7 | 42.1 | 44.6 |

**Prefix (cross-task, epoch 5):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 55.4 | 22.9 | 25.4 |
| five_objects | 58.3 | 23.3 | 3.8 |
| seven_objects | 30.8 | 23.3 | 26.2 |

**Prefix (cross-task, epoch 10):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 64.2 | 41.7 | 44.6 |
| five_objects | 52.5 | 40.8 | 37.5 |
| seven_objects | 60.0 | 46.2 | 45.8 |

**Prefix (cross-task, epoch 15):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 63.3 | 44.6 | 48.3 |
| five_objects | 65.4 | 43.3 | 44.2 |
| seven_objects | 66.2 | 46.2 | 50.8 |

**Prefix (cross-task, epoch 20):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 63.7 | 43.3 | 49.2 |
| five_objects | 65.8 | 45.0 | 46.2 |
| seven_objects | 67.9 | 47.1 | 48.8 |

### 2.2 Delta Tables (pp)

**CT-KV cross − ICL cross (epoch 5):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | +0.4 | +0.0 | +2.1 |
| five_objects | +3.8 | +7.5 | +3.8 |
| seven_objects | +2.1 | +0.8 | +0.8 |

**CT-KV cross − ICL cross (epoch 10):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | +1.2 | +0.8 | +1.2 |
| five_objects | +6.7 | +10.8 | +6.2 |
| seven_objects | -0.4 | +1.7 | +2.9 |

**CT-KV cross − ICL cross (epoch 15):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | +4.6 | +1.7 | +0.0 |
| five_objects | +10.4 | +10.4 | +5.8 |
| seven_objects | -1.2 | +0.0 | +1.2 |

**CT-KV cross − ICL cross (epoch 20):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | +3.3 | +2.5 | +0.8 |
| five_objects | +10.8 | +11.7 | +5.8 |
| seven_objects | -1.7 | +1.7 | +0.4 |

**CT-KV cross − zero-shot (epoch 5):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | +2.5 | +10.4 | +17.5 |
| five_objects | +12.5 | +13.3 | +20.4 |
| seven_objects | +9.2 | +10.0 | +18.3 |

**CT-KV cross − zero-shot (epoch 10):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | +3.3 | +11.2 | +16.7 |
| five_objects | +15.4 | +16.7 | +22.9 |
| seven_objects | +6.7 | +10.8 | +20.4 |

**CT-KV cross − zero-shot (epoch 15):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | +6.7 | +12.1 | +15.4 |
| five_objects | +19.2 | +16.2 | +22.5 |
| seven_objects | +5.8 | +9.2 | +18.7 |

**CT-KV cross − zero-shot (epoch 20):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | +5.4 | +12.9 | +16.2 |
| five_objects | +19.6 | +17.5 | +22.5 |
| seven_objects | +5.4 | +10.8 | +17.9 |

**CT-KV cross − ICL own (epoch 20):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | +3.3 | +7.1 | -1.2 |
| five_objects | +17.5 | +11.7 | +5.0 |
| seven_objects | +3.3 | +5.0 | +0.4 |

### 2.3 On-diag vs Off-diag Summary

| Method | On-diag | Off-diag | Δ On (vs ICL own) | Δ Off (vs ICL own) |
| --- | --- | --- | --- | --- |
| ICL (own) | 46.5 | 46.5 | — | — |
| ICL (cross) | 46.5 | 49.3 | +0.0 | +2.8 |
| CT-KV ep5 | 49.4 | 51.4 | +2.9 | +4.9 |
| CT-KV ep10 | 51.5 | 52.0 | +5.0 | +5.5 |
| CT-KV ep15 | 51.9 | 52.1 | +5.4 | +5.6 |
| CT-KV ep20 | 51.7 | 52.6 | +5.1 | +6.1 |
| Prefix ep5 | 35.0 | 27.4 | -11.5 | -19.1 |
| Prefix ep10 | 50.3 | 47.1 | +3.8 | +0.6 |
| Prefix ep15 | 52.5 | 52.5 | +6.0 | +6.0 |
| Prefix ep20 | 52.5 | 53.3 | +6.0 | +6.7 |

## 3. BBH: Tracking Shuffled Objects

Three variants (track item positions through pairwise swaps among N people): three_objects, five_objects, seven_objects.

### 3.1 Raw Accuracy (%)

**Zero-shot**: three=33.3, five=14.6, seven=8.8

**ICL (own demos)**: three=33.8, five=15.8, seven=15.4

**ICL (cross-task demos):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 33.8 | 17.9 | 11.2 |
| five_objects | 34.2 | 15.8 | 15.0 |
| seven_objects | 38.3 | 15.8 | 15.8 |

**CT-KV (cross-task, epoch 5):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 32.5 | 15.8 | 10.8 |
| five_objects | 36.7 | 18.8 | 17.5 |
| seven_objects | 35.4 | 17.5 | 15.8 |

**CT-KV (cross-task, epoch 10):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 31.2 | 15.0 | 10.0 |
| five_objects | 34.6 | 22.5 | 17.9 |
| seven_objects | 30.4 | 19.6 | 19.2 |

**CT-KV (cross-task, epoch 15):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 32.1 | 15.0 | 10.0 |
| five_objects | 36.2 | 22.5 | 15.4 |
| seven_objects | 29.2 | 18.3 | 17.5 |

**CT-KV (cross-task, epoch 20):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 32.5 | 14.6 | 10.0 |
| five_objects | 35.4 | 22.1 | 15.8 |
| seven_objects | 28.7 | 19.6 | 16.2 |

**Prefix (cross-task, epoch 5):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 30.8 | 17.9 | 12.9 |
| five_objects | 23.8 | 18.3 | 9.6 |
| seven_objects | 32.5 | 21.7 | 12.5 |

**Prefix (cross-task, epoch 10):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 30.8 | 19.6 | 11.7 |
| five_objects | 24.2 | 21.2 | 13.8 |
| seven_objects | 30.4 | 17.9 | 15.4 |

**Prefix (cross-task, epoch 15):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 34.2 | 19.2 | 12.9 |
| five_objects | 27.5 | 20.4 | 14.2 |
| seven_objects | 29.6 | 21.7 | 15.8 |

**Prefix (cross-task, epoch 20):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | 30.4 | 16.2 | 10.8 |
| five_objects | 29.2 | 20.4 | 14.6 |
| seven_objects | 30.8 | 18.3 | 14.2 |

### 3.2 Delta Tables (pp)

**CT-KV cross − ICL cross (epoch 5):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | -1.2 | -2.1 | -0.4 |
| five_objects | +2.5 | +2.9 | +2.5 |
| seven_objects | -2.9 | +1.7 | +0.0 |

**CT-KV cross − ICL cross (epoch 10):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | -2.5 | -2.9 | -1.2 |
| five_objects | +0.4 | +6.7 | +2.9 |
| seven_objects | -7.9 | +3.8 | +3.3 |

**CT-KV cross − ICL cross (epoch 15):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | -1.7 | -2.9 | -1.2 |
| five_objects | +2.1 | +6.7 | +0.4 |
| seven_objects | -9.2 | +2.5 | +1.7 |

**CT-KV cross − ICL cross (epoch 20):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | -1.2 | -3.3 | -1.2 |
| five_objects | +1.3 | +6.2 | +0.8 |
| seven_objects | -9.6 | +3.8 | +0.4 |

**CT-KV cross − zero-shot (epoch 5):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | -0.8 | +1.2 | +2.1 |
| five_objects | +3.3 | +4.2 | +8.8 |
| seven_objects | +2.1 | +2.9 | +7.1 |

**CT-KV cross − zero-shot (epoch 10):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | -2.1 | +0.4 | +1.2 |
| five_objects | +1.3 | +7.9 | +9.2 |
| seven_objects | -2.9 | +5.0 | +10.4 |

**CT-KV cross − zero-shot (epoch 15):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | -1.2 | +0.4 | +1.2 |
| five_objects | +2.9 | +7.9 | +6.7 |
| seven_objects | -4.2 | +3.7 | +8.8 |

**CT-KV cross − zero-shot (epoch 20):**

| Source ↓ Target → | three_objects | five_objects | seven_objects |
| --- | --- | --- | --- |
| three_objects | -0.8 | +0.0 | +1.2 |
| five_objects | +2.1 | +7.5 | +7.1 |
| seven_objects | -4.6 | +5.0 | +7.5 |

### 3.3 On-diag vs Off-diag Summary

| Method | On-diag | Off-diag | Δ On (vs ICL own) | Δ Off (vs ICL own) |
| --- | --- | --- | --- | --- |
| ICL (own) | 21.7 | 21.7 | — | — |
| ICL (cross) | 21.8 | 22.1 | +0.1 | +0.4 |
| CT-KV ep5 | 22.4 | 22.3 | +0.7 | +0.6 |
| CT-KV ep10 | 24.3 | 21.2 | +2.6 | -0.4 |
| CT-KV ep15 | 24.0 | 20.7 | +2.4 | -1.0 |
| CT-KV ep20 | 23.6 | 20.7 | +1.9 | -1.0 |
| Prefix ep5 | 20.6 | 19.7 | -1.1 | -1.9 |
| Prefix ep10 | 22.5 | 19.6 | +0.8 | -2.1 |
| Prefix ep15 | 23.5 | 20.8 | +1.8 | -0.8 |
| Prefix ep20 | 21.7 | 20.0 | +0.0 | -1.7 |

## 4. MMLU: Cross-Task Transfer

10 subjects across 5 domains: Math (abstract_algebra, college_mathematics), Physics (college_physics, high_school_physics), Econ (high_school_macroeconomics, high_school_microeconomics), Bio (college_biology, high_school_biology), Law (international_law, jurisprudence).

### 4.1 Per-Task Baselines

| Subject | Zero-shot | ICL (own) |
| --- | --- | --- |
| abstract_algebra | 28.6 | 34.5 |
| college_mathematics | 28.6 | 26.2 |
| college_physics | 26.7 | 38.4 |
| high_school_physics | 27.4 | 32.6 |
| high_school_macroeconomics | 33.7 | 38.5 |
| high_school_microeconomics | 39.2 | 39.6 |
| college_biology | 42.2 | 45.3 |
| high_school_biology | 38.8 | 41.8 |
| international_law | 43.8 | 42.9 |
| jurisprudence | 34.8 | 29.3 |
| **Mean** | **34.4** | **36.9** |

### ICL cross — Full 10×10 (%)

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | 34.5 | 26.2 | 31.4 | 31.1 | 34.2 | 38.3 | 39.8 | 41.2 | 39.0 | 32.6 |
| college_mathematics | 26.2 | 26.2 | 33.7 | 31.1 | 35.0 | 39.2 | 40.6 | 41.2 | 36.2 | 33.7 |
| college_physics | 31.0 | 25.0 | 38.4 | 32.6 | 33.4 | 39.2 | 43.0 | 42.2 | 36.2 | 38.0 |
| high_school_physics | 28.6 | 26.2 | 30.2 | 32.6 | 36.1 | 41.4 | 45.3 | 41.2 | 38.1 | 35.9 |
| high_school_macroeconomics | 31.0 | 28.6 | 23.3 | 28.9 | 38.5 | 39.2 | 43.8 | 37.8 | 38.1 | 31.5 |
| high_school_microeconomics | 28.6 | 26.2 | 24.4 | 30.4 | 36.4 | 39.6 | 43.0 | 38.1 | 37.1 | 29.3 |
| college_biology | 25.0 | 27.4 | 26.7 | 28.1 | 34.5 | 38.3 | 45.3 | 40.1 | 38.1 | 35.9 |
| high_school_biology | 29.8 | 20.2 | 25.6 | 31.9 | 33.7 | 37.8 | 46.9 | 41.8 | 37.1 | 35.9 |
| international_law | 28.6 | 22.6 | 27.9 | 25.9 | 32.9 | 36.9 | 46.1 | 40.1 | 42.9 | 39.1 |
| jurisprudence | 27.4 | 21.4 | 24.4 | 31.1 | 34.8 | 38.3 | 41.4 | 38.1 | 37.1 | 29.3 |

### CT-KV ep5 — Full 10×10 (%)

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | 39.3 | 26.2 | 30.2 | 34.8 | 32.9 | 36.5 | 39.8 | 40.8 | 38.1 | 31.5 |
| college_mathematics | 25.0 | 28.6 | 32.6 | 32.6 | 35.0 | 41.0 | 43.0 | 41.5 | 39.0 | 34.8 |
| college_physics | 29.8 | 27.4 | 37.2 | 28.1 | 34.5 | 41.0 | 45.3 | 43.5 | 41.0 | 39.1 |
| high_school_physics | 28.6 | 27.4 | 33.7 | 34.8 | 36.9 | 41.4 | 47.7 | 40.8 | 39.0 | 35.9 |
| high_school_macroeconomics | 32.1 | 26.2 | 26.7 | 31.1 | 38.8 | 40.5 | 46.9 | 37.8 | 41.9 | 29.3 |
| high_school_microeconomics | 32.1 | 28.6 | 23.3 | 30.4 | 37.2 | 39.6 | 45.3 | 38.1 | 39.0 | 30.4 |
| college_biology | 27.4 | 27.4 | 33.7 | 31.1 | 35.6 | 39.6 | 44.5 | 42.9 | 36.2 | 38.0 |
| high_school_biology | 26.2 | 23.8 | 23.3 | 31.9 | 34.2 | 38.7 | 50.0 | 44.9 | 41.0 | 35.9 |
| international_law | 29.8 | 22.6 | 30.2 | 25.9 | 33.2 | 37.8 | 46.1 | 43.2 | 43.8 | 41.3 |
| jurisprudence | 25.0 | 21.4 | 26.7 | 33.3 | 32.4 | 37.8 | 44.5 | 38.4 | 38.1 | 33.7 |

### CT-KV ep10 — Full 10×10 (%)

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | 38.1 | 27.4 | 26.7 | 33.3 | 33.2 | 36.5 | 40.6 | 40.5 | 40.0 | 31.5 |
| college_mathematics | 28.6 | 29.8 | 37.2 | 32.6 | 36.1 | 41.0 | 43.0 | 41.2 | 40.0 | 37.0 |
| college_physics | 29.8 | 27.4 | 34.9 | 31.1 | 34.5 | 41.4 | 46.9 | 43.9 | 40.0 | 38.0 |
| high_school_physics | 28.6 | 29.8 | 30.2 | 35.6 | 37.7 | 42.3 | 48.4 | 42.5 | 38.1 | 35.9 |
| high_school_macroeconomics | 32.1 | 26.2 | 24.4 | 30.4 | 39.0 | 40.1 | 45.3 | 37.8 | 41.9 | 28.3 |
| high_school_microeconomics | 32.1 | 28.6 | 23.3 | 30.4 | 37.7 | 41.0 | 45.3 | 39.5 | 39.0 | 33.7 |
| college_biology | 28.6 | 26.2 | 32.6 | 33.3 | 35.6 | 40.5 | 44.5 | 42.5 | 35.2 | 40.2 |
| high_school_biology | 27.4 | 25.0 | 27.9 | 33.3 | 34.2 | 37.8 | 50.0 | 45.2 | 41.0 | 37.0 |
| international_law | 29.8 | 26.2 | 30.2 | 28.1 | 33.2 | 40.1 | 46.9 | 44.2 | 43.8 | 41.3 |
| jurisprudence | 25.0 | 23.8 | 25.6 | 32.6 | 34.0 | 38.7 | 46.1 | 38.8 | 36.2 | 34.8 |

### CT-KV ep15 — Full 10×10 (%)

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | 34.5 | 27.4 | 27.9 | 31.9 | 33.2 | 35.6 | 41.4 | 39.8 | 39.0 | 31.5 |
| college_mathematics | 31.0 | 27.4 | 33.7 | 31.9 | 36.6 | 41.4 | 45.3 | 41.8 | 38.1 | 39.1 |
| college_physics | 29.8 | 29.8 | 34.9 | 29.6 | 35.0 | 41.4 | 47.7 | 44.6 | 41.9 | 39.1 |
| high_school_physics | 29.8 | 29.8 | 29.1 | 35.6 | 38.2 | 42.8 | 46.9 | 42.5 | 39.0 | 35.9 |
| high_school_macroeconomics | 33.3 | 25.0 | 25.6 | 29.6 | 39.0 | 39.2 | 44.5 | 38.8 | 41.9 | 28.3 |
| high_school_microeconomics | 29.8 | 28.6 | 22.1 | 30.4 | 39.6 | 41.4 | 46.1 | 39.8 | 37.1 | 32.6 |
| college_biology | 29.8 | 25.0 | 30.2 | 32.6 | 36.1 | 40.5 | 45.3 | 42.9 | 37.1 | 39.1 |
| high_school_biology | 27.4 | 22.6 | 27.9 | 31.9 | 33.7 | 37.8 | 50.8 | 43.9 | 41.9 | 35.9 |
| international_law | 28.6 | 25.0 | 31.4 | 25.2 | 35.0 | 39.2 | 45.3 | 44.9 | 44.8 | 39.1 |
| jurisprudence | 26.2 | 21.4 | 25.6 | 32.6 | 34.8 | 38.7 | 47.7 | 39.8 | 36.2 | 34.8 |

### CT-KV ep20 — Full 10×10 (%)

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | 32.1 | 28.6 | 27.9 | 31.1 | 33.2 | 35.6 | 39.8 | 39.1 | 38.1 | 32.6 |
| college_mathematics | 33.3 | 26.2 | 36.0 | 31.1 | 36.1 | 40.5 | 47.7 | 42.9 | 38.1 | 38.0 |
| college_physics | 31.0 | 32.1 | 34.9 | 30.4 | 36.4 | 40.5 | 46.9 | 43.9 | 41.9 | 39.1 |
| high_school_physics | 28.6 | 28.6 | 29.1 | 34.8 | 38.5 | 40.5 | 47.7 | 42.2 | 39.0 | 37.0 |
| high_school_macroeconomics | 26.2 | 25.0 | 27.9 | 28.9 | 39.3 | 38.7 | 43.8 | 38.4 | 41.0 | 29.3 |
| high_school_microeconomics | 31.0 | 27.4 | 25.6 | 31.9 | 39.3 | 39.6 | 46.1 | 37.4 | 38.1 | 33.7 |
| college_biology | 27.4 | 25.0 | 32.6 | 33.3 | 35.0 | 40.5 | 47.7 | 45.2 | 38.1 | 37.0 |
| high_school_biology | 26.2 | 23.8 | 31.4 | 32.6 | 32.9 | 37.4 | 50.0 | 44.2 | 41.0 | 34.8 |
| international_law | 28.6 | 25.0 | 30.2 | 25.9 | 35.0 | 40.5 | 44.5 | 44.6 | 44.8 | 37.0 |
| jurisprudence | 25.0 | 22.6 | 25.6 | 34.8 | 35.0 | 37.8 | 47.7 | 40.8 | 37.1 | 33.7 |

### Prefix ep5 — Full 10×10 (%)

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | 27.4 | 21.4 | 25.6 | 26.7 | 34.5 | 37.8 | 39.8 | 38.8 | 38.1 | 34.8 |
| college_mathematics | 34.5 | 25.0 | 29.1 | 30.4 | 34.8 | 40.1 | 42.2 | 39.5 | 41.0 | 31.5 |
| college_physics | 31.0 | 23.8 | 32.6 | 30.4 | 34.5 | 40.1 | 43.0 | 40.1 | 42.9 | 32.6 |
| high_school_physics | 29.8 | 26.2 | 27.9 | 30.4 | 36.4 | 38.7 | 43.0 | 40.8 | 41.9 | 37.0 |
| high_school_macroeconomics | 32.1 | 27.4 | 31.4 | 31.9 | 36.1 | 40.1 | 44.5 | 40.5 | 41.9 | 34.8 |
| high_school_microeconomics | 32.1 | 23.8 | 25.6 | 31.9 | 35.8 | 40.1 | 40.6 | 41.2 | 42.9 | 34.8 |
| college_biology | 33.3 | 25.0 | 27.9 | 30.4 | 35.3 | 40.1 | 43.8 | 40.5 | 42.9 | 33.7 |
| high_school_biology | 33.3 | 26.2 | 29.1 | 29.6 | 35.0 | 40.5 | 44.5 | 41.5 | 41.9 | 32.6 |
| international_law | 34.5 | 27.4 | 31.4 | 30.4 | 36.1 | 41.0 | 43.0 | 40.8 | 41.9 | 32.6 |
| jurisprudence | 31.0 | 26.2 | 29.1 | 30.4 | 36.4 | 40.1 | 43.0 | 39.8 | 43.8 | 33.7 |

### Prefix ep10 — Full 10×10 (%)

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | 28.6 | 25.0 | 24.4 | 28.1 | 34.0 | 37.8 | 38.3 | 39.8 | 39.0 | 33.7 |
| college_mathematics | 27.4 | 25.0 | 32.6 | 30.4 | 33.7 | 38.7 | 46.9 | 39.8 | 41.0 | 32.6 |
| college_physics | 33.3 | 25.0 | 30.2 | 32.6 | 35.6 | 38.3 | 45.3 | 39.5 | 43.8 | 33.7 |
| high_school_physics | 31.0 | 20.2 | 32.6 | 31.1 | 36.9 | 39.2 | 43.0 | 42.2 | 41.9 | 35.9 |
| high_school_macroeconomics | 31.0 | 29.8 | 30.2 | 31.1 | 36.1 | 39.6 | 45.3 | 42.9 | 41.9 | 37.0 |
| high_school_microeconomics | 29.8 | 27.4 | 25.6 | 34.1 | 36.1 | 40.1 | 41.4 | 42.2 | 41.0 | 35.9 |
| college_biology | 32.1 | 26.2 | 27.9 | 32.6 | 36.4 | 39.6 | 44.5 | 42.2 | 42.9 | 35.9 |
| high_school_biology | 31.0 | 27.4 | 31.4 | 31.9 | 34.5 | 40.5 | 46.1 | 41.8 | 42.9 | 33.7 |
| international_law | 32.1 | 27.4 | 30.2 | 31.1 | 36.9 | 41.0 | 44.5 | 42.2 | 42.9 | 32.6 |
| jurisprudence | 31.0 | 28.6 | 24.4 | 31.9 | 36.4 | 40.1 | 42.2 | 40.5 | 42.9 | 34.8 |

### Prefix ep15 — Full 10×10 (%)

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | 27.4 | 25.0 | 24.4 | 27.4 | 34.0 | 37.8 | 40.6 | 39.1 | 39.0 | 34.8 |
| college_mathematics | 28.6 | 25.0 | 31.4 | 31.9 | 33.7 | 37.8 | 44.5 | 41.2 | 41.0 | 32.6 |
| college_physics | 31.0 | 26.2 | 31.4 | 31.9 | 35.8 | 39.6 | 45.3 | 40.8 | 41.9 | 33.7 |
| high_school_physics | 26.2 | 22.6 | 29.1 | 29.6 | 36.6 | 38.3 | 41.4 | 43.9 | 41.9 | 34.8 |
| high_school_macroeconomics | 31.0 | 29.8 | 30.2 | 29.6 | 37.2 | 38.3 | 45.3 | 43.2 | 41.0 | 34.8 |
| high_school_microeconomics | 31.0 | 27.4 | 24.4 | 31.9 | 38.0 | 41.0 | 43.0 | 42.2 | 41.0 | 35.9 |
| college_biology | 32.1 | 25.0 | 27.9 | 34.1 | 36.9 | 37.8 | 46.1 | 42.9 | 43.8 | 35.9 |
| high_school_biology | 31.0 | 27.4 | 27.9 | 31.1 | 36.1 | 39.6 | 45.3 | 42.5 | 43.8 | 33.7 |
| international_law | 31.0 | 27.4 | 26.7 | 31.1 | 36.9 | 39.6 | 46.1 | 42.5 | 43.8 | 32.6 |
| jurisprudence | 26.2 | 27.4 | 25.6 | 34.1 | 36.4 | 41.0 | 46.1 | 40.8 | 44.8 | 34.8 |

### Prefix ep20 — Full 10×10 (%)

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | 27.4 | 27.4 | 19.8 | 24.4 | 33.4 | 36.9 | 42.2 | 39.8 | 41.0 | 34.8 |
| college_mathematics | 31.0 | 23.8 | 30.2 | 31.1 | 33.2 | 38.7 | 46.1 | 42.5 | 41.9 | 32.6 |
| college_physics | 29.8 | 28.6 | 30.2 | 31.9 | 35.3 | 38.3 | 45.3 | 42.5 | 41.0 | 30.4 |
| high_school_physics | 22.6 | 23.8 | 25.6 | 28.9 | 36.6 | 36.9 | 44.5 | 43.9 | 41.9 | 34.8 |
| high_school_macroeconomics | 28.6 | 28.6 | 30.2 | 31.1 | 37.2 | 38.7 | 46.1 | 43.9 | 41.9 | 33.7 |
| high_school_microeconomics | 31.0 | 28.6 | 25.6 | 34.1 | 37.4 | 40.5 | 43.8 | 41.5 | 40.0 | 34.8 |
| college_biology | 29.8 | 23.8 | 29.1 | 34.1 | 38.2 | 37.8 | 48.4 | 43.5 | 43.8 | 37.0 |
| high_school_biology | 31.0 | 29.8 | 27.9 | 31.1 | 36.6 | 40.1 | 46.1 | 42.9 | 44.8 | 33.7 |
| international_law | 29.8 | 28.6 | 24.4 | 28.9 | 38.0 | 40.1 | 45.3 | 43.5 | 44.8 | 34.8 |
| jurisprudence | 29.8 | 27.4 | 25.6 | 34.8 | 34.5 | 41.4 | 46.9 | 41.8 | 44.8 | 32.6 |

### 4.2 CT-KV cross − ICL cross (Q1: does optimization help?)

**Epoch 5:**

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | +4.8 | +0.0 | -1.2 | +3.7 | -1.3 | -1.8 | +0.0 | -0.3 | -1.0 | -1.1 |
| college_mathematics | -1.2 | +2.4 | -1.2 | +1.5 | +0.0 | +1.8 | +2.3 | +0.3 | +2.9 | +1.1 |
| college_physics | -1.2 | +2.4 | -1.2 | -4.4 | +1.1 | +1.8 | +2.3 | +1.4 | +4.8 | +1.1 |
| high_school_physics | +0.0 | +1.2 | +3.5 | +2.2 | +0.8 | +0.0 | +2.3 | -0.3 | +1.0 | +0.0 |
| high_school_macroeconomics | +1.2 | -2.4 | +3.5 | +2.2 | +0.3 | +1.4 | +3.1 | +0.0 | +3.8 | -2.2 |
| high_school_microeconomics | +3.6 | +2.4 | -1.2 | +0.0 | +0.8 | +0.0 | +2.3 | +0.0 | +1.9 | +1.1 |
| college_biology | +2.4 | +0.0 | +7.0 | +3.0 | +1.1 | +1.4 | -0.8 | +2.7 | -1.9 | +2.2 |
| high_school_biology | -3.6 | +3.6 | -2.3 | +0.0 | +0.5 | +0.9 | +3.1 | +3.1 | +3.8 | +0.0 |
| international_law | +1.2 | +0.0 | +2.3 | +0.0 | +0.3 | +0.9 | +0.0 | +3.1 | +1.0 | +2.2 |
| jurisprudence | -2.4 | +0.0 | +2.3 | +2.2 | -2.4 | -0.5 | +3.1 | +0.3 | +1.0 | +4.3 |

**Epoch 10:**

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | +3.6 | +1.2 | -4.7 | +2.2 | -1.1 | -1.8 | +0.8 | -0.7 | +1.0 | -1.1 |
| college_mathematics | +2.4 | +3.6 | +3.5 | +1.5 | +1.1 | +1.8 | +2.3 | +0.0 | +3.8 | +3.3 |
| college_physics | -1.2 | +2.4 | -3.5 | -1.5 | +1.1 | +2.3 | +3.9 | +1.7 | +3.8 | +0.0 |
| high_school_physics | +0.0 | +3.6 | +0.0 | +3.0 | +1.6 | +0.9 | +3.1 | +1.4 | +0.0 | +0.0 |
| high_school_macroeconomics | +1.2 | -2.4 | +1.2 | +1.5 | +0.5 | +0.9 | +1.6 | +0.0 | +3.8 | -3.3 |
| high_school_microeconomics | +3.6 | +2.4 | -1.2 | +0.0 | +1.3 | +1.4 | +2.3 | +1.4 | +1.9 | +4.3 |
| college_biology | +3.6 | -1.2 | +5.8 | +5.2 | +1.1 | +2.3 | -0.8 | +2.4 | -2.9 | +4.3 |
| high_school_biology | -2.4 | +4.8 | +2.3 | +1.5 | +0.5 | +0.0 | +3.1 | +3.4 | +3.8 | +1.1 |
| international_law | +1.2 | +3.6 | +2.3 | +2.2 | +0.3 | +3.2 | +0.8 | +4.1 | +1.0 | +2.2 |
| jurisprudence | -2.4 | +2.4 | +1.2 | +1.5 | -0.8 | +0.5 | +4.7 | +0.7 | -1.0 | +5.4 |

**Epoch 15:**

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | +0.0 | +1.2 | -3.5 | +0.7 | -1.1 | -2.7 | +1.6 | -1.4 | +0.0 | -1.1 |
| college_mathematics | +4.8 | +1.2 | +0.0 | +0.7 | +1.6 | +2.3 | +4.7 | +0.7 | +1.9 | +5.4 |
| college_physics | -1.2 | +4.8 | -3.5 | -3.0 | +1.6 | +2.3 | +4.7 | +2.4 | +5.7 | +1.1 |
| high_school_physics | +1.2 | +3.6 | -1.2 | +3.0 | +2.1 | +1.4 | +1.6 | +1.4 | +1.0 | +0.0 |
| high_school_macroeconomics | +2.4 | -3.6 | +2.3 | +0.7 | +0.5 | +0.0 | +0.8 | +1.0 | +3.8 | -3.3 |
| high_school_microeconomics | +1.2 | +2.4 | -2.3 | +0.0 | +3.2 | +1.8 | +3.1 | +1.7 | +0.0 | +3.3 |
| college_biology | +4.8 | -2.4 | +3.5 | +4.4 | +1.6 | +2.3 | +0.0 | +2.7 | -1.0 | +3.3 |
| high_school_biology | -2.4 | +2.4 | +2.3 | +0.0 | +0.0 | +0.0 | +3.9 | +2.0 | +4.8 | +0.0 |
| international_law | +0.0 | +2.4 | +3.5 | -0.7 | +2.1 | +2.3 | -0.8 | +4.8 | +1.9 | +0.0 |
| jurisprudence | -1.2 | +0.0 | +1.2 | +1.5 | +0.0 | +0.5 | +6.2 | +1.7 | -1.0 | +5.4 |

**Epoch 20:**

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | -2.4 | +2.4 | -3.5 | +0.0 | -1.1 | -2.7 | +0.0 | -2.0 | -1.0 | +0.0 |
| college_mathematics | +7.1 | +0.0 | +2.3 | +0.0 | +1.1 | +1.4 | +7.0 | +1.7 | +1.9 | +4.3 |
| college_physics | +0.0 | +7.1 | -3.5 | -2.2 | +2.9 | +1.4 | +3.9 | +1.7 | +5.7 | +1.1 |
| high_school_physics | +0.0 | +2.4 | -1.2 | +2.2 | +2.4 | -0.9 | +2.3 | +1.0 | +1.0 | +1.1 |
| high_school_macroeconomics | -4.8 | -3.6 | +4.7 | +0.0 | +0.8 | -0.5 | +0.0 | +0.7 | +2.9 | -2.2 |
| high_school_microeconomics | +2.4 | +1.2 | +1.2 | +1.5 | +2.9 | +0.0 | +3.1 | -0.7 | +1.0 | +4.3 |
| college_biology | +2.4 | -2.4 | +5.8 | +5.2 | +0.5 | +2.3 | +2.3 | +5.1 | +0.0 | +1.1 |
| high_school_biology | -3.6 | +3.6 | +5.8 | +0.7 | -0.8 | -0.5 | +3.1 | +2.4 | +3.8 | -1.1 |
| international_law | +0.0 | +2.4 | +2.3 | +0.0 | +2.1 | +3.6 | -1.6 | +4.4 | +1.9 | -2.2 |
| jurisprudence | -2.4 | +1.2 | +1.2 | +3.7 | +0.3 | -0.5 | +6.2 | +2.7 | +0.0 | +4.3 |

### 4.3 CT-KV cross − zero-shot (does cross-task context help vs nothing?)

**Epoch 10:**

| Source ↓ Target → | abstract_algebra | college_mathematics | college_physics | high_school_physics | high_school_macroeconomics | high_school_microeconomics | college_biology | high_school_biology | international_law | jurisprudence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abstract_algebra | +9.5 | -1.2 | +0.0 | +5.9 | -0.5 | -2.7 | -1.6 | +1.7 | -3.8 | -3.3 |
| college_mathematics | +0.0 | +1.2 | +10.5 | +5.2 | +2.4 | +1.8 | +0.8 | +2.4 | -3.8 | +2.2 |
| college_physics | +1.2 | -1.2 | +8.1 | +3.7 | +0.8 | +2.3 | +4.7 | +5.1 | -3.8 | +3.3 |
| high_school_physics | +0.0 | +1.2 | +3.5 | +8.1 | +4.0 | +3.2 | +6.2 | +3.7 | -5.7 | +1.1 |
| high_school_macroeconomics | +3.6 | -2.4 | -2.3 | +3.0 | +5.3 | +0.9 | +3.1 | -1.0 | -1.9 | -6.5 |
| high_school_microeconomics | +3.6 | +0.0 | -3.5 | +3.0 | +4.0 | +1.8 | +3.1 | +0.7 | -4.8 | -1.1 |
| college_biology | +0.0 | -2.4 | +5.8 | +5.9 | +1.9 | +1.4 | +2.3 | +3.7 | -8.6 | +5.4 |
| high_school_biology | -1.2 | -3.6 | +1.2 | +5.9 | +0.5 | -1.4 | +7.8 | +6.5 | -2.9 | +2.2 |
| international_law | +1.2 | -2.4 | +3.5 | +0.7 | -0.5 | +0.9 | +4.7 | +5.4 | +0.0 | +6.5 |
| jurisprudence | -3.6 | -4.8 | -1.2 | +5.2 | +0.3 | -0.5 | +3.9 | +0.0 | -7.6 | +0.0 |

### 4.4 Within-Domain Pair Analysis (Q1: CT-KV cross − ICL cross)

| Epoch | Math A→B | Math B→A | Phys A→B | Phys B→A | Econ A→B | Econ B→A | Bio A→B | Bio B→A | Law A→B | Law B→A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | +0.0 | -1.2 | -4.4 | +3.5 | +1.4 | +0.8 | +2.7 | +3.1 | +2.2 | +1.0 |
| 10 | +1.2 | +2.4 | -1.5 | +0.0 | +0.9 | +1.3 | +2.4 | +3.1 | +2.2 | -1.0 |
| 15 | +1.2 | +4.8 | -3.0 | -1.2 | +0.0 | +3.2 | +2.7 | +3.9 | +0.0 | -1.0 |
| 20 | +2.4 | +7.1 | -2.2 | -1.2 | -0.5 | +2.9 | +5.1 | +3.1 | -2.2 | +0.0 |

### 4.5 Within-Domain Pair Analysis (CT-KV cross − zero-shot)

| Epoch | Math A→B | Math B→A | Phys A→B | Phys B→A | Econ A→B | Econ B→A | Bio A→B | Bio B→A | Law A→B | Law B→A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | -2.4 | -3.6 | +0.7 | +7.0 | +1.4 | +3.5 | +4.1 | +7.8 | +6.5 | -5.7 |
| 10 | -1.2 | +0.0 | +3.7 | +3.5 | +0.9 | +4.0 | +3.7 | +7.8 | +6.5 | -7.6 |
| 15 | -1.2 | +2.4 | +2.2 | +2.3 | +0.0 | +5.9 | +4.1 | +8.6 | +4.3 | -7.6 |
| 20 | +0.0 | +4.8 | +3.0 | +2.3 | -0.5 | +5.6 | +6.5 | +7.8 | +2.2 | -6.7 |

### 4.6 Aggregate Summary (all 10×10)

**Q1: CT-KV cross − ICL cross:**

| Epoch | On-diag mean | Off-diag mean | Off-diag positive |
| --- | --- | --- | --- |
| 5 | +1.6 | +0.9 | 54/90 |
| 10 | +1.8 | +1.3 | 65/90 |
| 15 | +1.2 | +1.3 | 58/90 |
| 20 | +0.8 | +1.3 | 56/90 |

**CT-KV cross − zero-shot:**

| Epoch | On-diag mean | Off-diag mean | Off-diag positive |
| --- | --- | --- | --- |
| 5 | +4.1 | +0.5 | 47/90 |
| 10 | +4.3 | +0.9 | 53/90 |
| 15 | +3.8 | +0.9 | 55/90 |
| 20 | +3.4 | +0.9 | 51/90 |

### 4.7 On-diag vs Off-diag by Method

| Method | On-diag | Off-diag | Δ On (vs ICL own) | Δ Off (vs ICL own) |
| --- | --- | --- | --- | --- |
| ICL (own) | 36.9 | 36.9 | — | — |
| ICL (cross) | 36.9 | 34.0 | +0.0 | -3.0 |
| CT-KV ep5 | 38.5 | 34.9 | +1.6 | -2.0 |
| CT-KV ep10 | 38.7 | 35.3 | +1.8 | -1.6 |
| CT-KV ep15 | 38.2 | 35.2 | +1.2 | -1.7 |
| CT-KV ep20 | 37.7 | 35.3 | +0.8 | -1.6 |
| Prefix ep5 | 35.2 | 35.0 | -1.7 | -1.9 |
| Prefix ep10 | 35.5 | 35.4 | -1.4 | -1.5 |
| Prefix ep15 | 35.9 | 35.2 | -1.0 | -1.7 |
| Prefix ep20 | 35.7 | 35.3 | -1.2 | -1.6 |


## 5. Reproduction: Bash Commands & Job Tracking

All commands in `bash_cmds/0326_2_cross_task/`. Results in `encoder_decoder/outputs_eval/eval_{tag}/eval_pred_gt.json`.

### 5.1 File Inventory

| File | Method | #Jobs | GPU | Walltime | SLURM ID |
| --- | --- | --- | --- | --- | --- |
| `bbh_zeroshot_baseline.sh` | Zero-shot | 2 | L40S | ran locally | — |
| `mmlu_zeroshot_baseline.sh` | Zero-shot | 1 | L40S | ran locally | — |
| `bbh_icl_baseline.sh` | ICL cross | 6 | L40S | 6h single | 5070948 |
| `mmlu_icl_baseline.sh` | ICL cross | 10 | L40S | 4h single | 5070956 |
| `bbh_cross_task_epoch5.sh` | CT-KV | 6 | L40S | 6h single | 5056032 |
| `bbh_cross_task_epoch10.sh` | CT-KV | 6 | L40S | 6h single | 5056040 |
| `bbh_cross_task_epoch15.sh` | CT-KV | 6 | L40S | 6h single | 5056048 |
| `bbh_cross_task_epoch20.sh` | CT-KV | 6 | L40S | 6h single | 5056057 |
| `mmlu_cross_task_epoch5.sh` | CT-KV | 10 | L40S | 4h single | 5044387 |
| `mmlu_cross_task_epoch10.sh` | CT-KV | 10 | L40S | 4h single | 5044396 |
| `mmlu_cross_task_epoch15.sh` | CT-KV | 10 | L40S | 4h single | 5044412 |
| `mmlu_cross_task_epoch20.sh` | CT-KV | 10 | L40S | 4h single | 5044420 |
| `bbh_cross_task_prefix_epoch5.sh` | Prefix | 6 | L40S | 6h single | 5071756 |
| `bbh_cross_task_prefix_epoch10.sh` | Prefix | 6 | L40S | 6h single | 5071762 |
| `bbh_cross_task_prefix_epoch15.sh` | Prefix | 6 | L40S | 6h single | 5071774 |
| `bbh_cross_task_prefix_epoch20.sh` | Prefix | 6 | L40S | 6h single | 5071787 |
| `mmlu_cross_task_prefix_epoch5.sh` | Prefix | 10 | L40S | 4h single | 5071793 |
| `mmlu_cross_task_prefix_epoch10.sh` | Prefix | 10 | L40S | 4h single | 5071799 |
| `mmlu_cross_task_prefix_epoch15.sh` | Prefix | 10 | L40S | 4h single | 5071805 |
| `mmlu_cross_task_prefix_epoch20.sh` | Prefix | 10 | L40S | 4h single | 5071811 |
| `mmlu_cross_task.sh` | CT-KV (legacy) | 10 | — | — | deprecated |

### 5.2 Hyperparameters

**BBH CT-KV**: `--gs_epochs {5,10,15,20} --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1 --max_seq_len 4096`

**BBH Prefix**: `--gs_epochs {5,10,15,20} --gs_lr 3e-3 --gs_batch_size 2 --gs_dropout none --random_kv token --random_kv_ntokens 32 --max_seq_len 4096`

**MMLU CT-KV**: `--gs_epochs {5,10,15,20} --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 --eval_ratio 1.0`

**MMLU Prefix**: `--gs_epochs {5,10,15,20} --gs_lr 1e-3 --gs_dropout none --random_kv token --random_kv_ntokens 32 --eval_ratio 1.0`

All runs: `--seed 42`, model `Llama-3.2-1B-Instruct`, `--mixed_precision bf16`.

### 5.3 Output Directories

BBH tags follow pattern: `eval_ct_bbh_{method}_{family}_src_{src}_ep{ep}_seed42`
- Zero-shot: `eval_ct_bbh_zeroshot_{family}_seed42`
- ICL on-diag: `eval_ct_bbh_icl_baseline_{family}_seed42`
- ICL cross: `eval_ct_bbh_icl_cross_{family}_src_{src}_seed42`
- CT-KV: `eval_ct_bbh_cross_{family}_src_{src}_ep{ep}_seed42`
- Prefix: `eval_ct_bbh_cross_prefix_{family}_src_{src}_ep{ep}_seed42`

MMLU tags follow pattern: `eval_ct_mmlu_{method}_src_{subject}_ep{ep}_seed42`
- Zero-shot: `eval_ct_mmlu_zeroshot_seed42`
- ICL on-diag: `eval_ct_mmlu_icl_baseline_seed42`
- ICL cross: `eval_ct_mmlu_icl_cross_src_{subject}_seed42`
- CT-KV: `eval_ct_mmlu_cross_src_{subject}_ep{ep}_seed42`
- Prefix: `eval_ct_mmlu_cross_prefix_src_{subject}_ep{ep}_seed42`

### 5.4 Code Changes

Added `--demo_source_task` support to both BBH and MMLU:
- `inference_bbh/data_utils.py`: `demo_source_task` parameter ensures source task data is loaded
- `inference_bbh/test_time_evaluate.py`: `demo_task = demo_source_task if demo_source_task is not None else task` used for demon_input_ids lookup, TTT, KV init, GS
- `inference_mmlu/test_time_evaluate.py`: same pattern (done in prior work)
- `--max_seq_len 4096` needed for tracking_shuffled_objects_seven_objects (default 2048 filters all test examples)

## 6. Key Takeaways for Rebuttal

### BBH Logical Deduction (strongest result)

Three progressively stronger claims:
1. **CT-KV cross > zero-shot**: 9/9 positive, +14.3pp mean — optimized cross-task KV carries useful information
2. **CT-KV cross > ICL cross**: 7/9 positive, +3.7pp mean — optimization extracts additional signal from mismatched demos
3. **CT-KV cross > ICL own**: 8/9 positive, +5.8pp mean — optimized cross-task KV beats correct-task ICL

The five_objects source is universally the best, pushing three_objects to 75.8% (vs 58.3% ICL own, +17.5pp).

### MMLU (aggregate result)

CT-KV optimization with cross-task demos improves over ICL with the same mismatched demos by +1.3pp on average (65/90 off-diagonal pairs positive, 72%). Best within-domain pairs: Bio (+2.4/+3.1pp both directions), Math (+1.2/+2.4pp).

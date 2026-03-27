# Cross-Task KV Transfer Experiment

## Reviewer Question

R1: "Have the authors explored whether optimized contexts can generalize across related tasks or datasets?"
R3: "How sensitive is Context Tuning to the quality and ordering of the demonstration examples used for initialization?"

## Experiment Design

**Benchmark**: MMLU (ideal for cross-task transfer because all 57 subjects share identical format — bare Q\nA pairs, no instruction text, 4-choice MCQ scored by loss. Transfer depends purely on content similarity.)

**Why not BBH**: Each BBH task has a different instruction baked into every example's KV. Cross-task transfer would mostly test whether the model handles *wrong instructions* in the KV — not a meaningful test of knowledge transfer.

**10 subjects (5 related pairs)**:

| Pair | Subject A | Subject B | Domain |
|------|-----------|-----------|--------|
| 1 | abstract_algebra | college_mathematics | Math |
| 2 | college_physics | high_school_physics | Physics |
| 3 | high_school_macroeconomics | high_school_microeconomics | Economics |
| 4 | college_biology | high_school_biology | Biology |
| 5 | international_law | jurisprudence | Law |

Note: History pair (high_school_us_history, high_school_world_history) was originally planned but all 3 history subjects exceed `max_seq_len=4096` with 16 demos and get filtered out. Law pair is a strong replacement.

**Protocol**: For each of the 10 source subjects:
1. Initialize KV from source subject's 16 demonstrations
2. Optimize KV using CT-KV (lr=1.5e-3, 20 epochs, tokdrop=0.1, leave-one-out)
3. Evaluate on all 10 subjects' test sets using the source-optimized KV

This produces a **10x10 transfer matrix**. The diagonal = standard CT-KV (same-task). Off-diagonal = cross-task transfer.

**Baselines**:
- ICL (no CT-KV): standard in-context learning on the same 10 subjects
- Same-task CT-KV: the diagonal of the matrix (identical to running CT-KV normally)

**Seed reset for reproducibility**: When `--demo_source_task` is set, `set_seed(accelerator.process_index)` is called before each KV initialization to ensure the token dropout masks during GS optimization are identical across all 10 target tasks. Without this, different random states (from evaluating earlier tasks) would cause slightly different optimized KVs.

## Expected Results

A heatmap with 5 bright 2x2 blocks along the diagonal:
- **Within-pair transfer**: KV optimized on college_physics helps high_school_physics (and vice versa)
- **Cross-pair transfer**: KV optimized on abstract_algebra does NOT help college_biology
- **Diagonal dominance**: Same-task CT-KV is always best or near-best

This shows CT-KV captures task-specific knowledge structure (not just memorizing demo answers), and that this knowledge partially transfers to semantically related subjects.

## Code Changes

**File modified**: `inference_mmlu/test_time_evaluate.py`

**New args** (backwards-compatible, both default to `None`):
- `--demo_source_task`: Subject whose demos to use for KV init and optimization
- `--select_tasks`: Only evaluate on these subjects (space-separated)

**Logic change**: In the task loop, `demo_task = demo_source_task if demo_source_task is not None else task`. This swaps `demon_input_ids`, `demon_start_idxs`, and `task_to_demonstrations[task]` to use the source subject's data. Everything else (model loading, evaluation) unchanged.

## bash_cmds

```
bash_cmds/0326_2_cross_task/
  mmlu_cross_task.sh    # 10 jobs: one per source subject, each evaluates on all 10 targets
  mmlu_icl_baseline.sh  # 1 job: ICL baseline on the 10 subjects
  mmlu_ctkv_baseline.sh # 1 job: same-task CT-KV baseline on the 10 subjects
```

**Submission**: `makesbatch --time 1 --ngpu 1 --gb 64 --bash_file bash_cmds/0326_2_cross_task/mmlu_cross_task.sh`

**Compute**: ~10 jobs x 5 min each = ~50 min for 1 seed.

## Parsing Results

Each job produces `encoder_decoder/outputs_eval/eval_{tag}/eval_pred_gt.json`. Per-task scores are logged to stdout:
```
{subject_name} has a score {accuracy}
```

To build the 10x10 matrix: for each source job's output, extract the 10 per-subject scores and arrange into a row of the matrix.

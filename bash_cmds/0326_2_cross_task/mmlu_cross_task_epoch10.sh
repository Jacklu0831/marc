# Cross-task KV transfer on MMLU: train KV on source subject, evaluate on 10 target subjects
# Addresses R1 Q1: "whether optimized contexts can generalize across related tasks"
# makesbatch --time 4 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0326_2_cross_task/mmlu_cross_task_epoch10.sh

# ct_mmlu_cross_src_abstract_algebra_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_abstract_algebra_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task abstract_algebra \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_cross_src_college_mathematics_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_college_mathematics_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task college_mathematics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_cross_src_college_physics_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_college_physics_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task college_physics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_cross_src_high_school_physics_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_high_school_physics_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task high_school_physics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_cross_src_high_school_macroeconomics_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_high_school_macroeconomics_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task high_school_macroeconomics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_cross_src_high_school_microeconomics_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_high_school_microeconomics_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task high_school_microeconomics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_cross_src_college_biology_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_college_biology_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task college_biology \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_cross_src_high_school_biology_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_high_school_biology_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task high_school_biology \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_cross_src_international_law_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_international_law_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task international_law \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_cross_src_jurisprudence_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_cross_src_jurisprudence_ep10_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --gs_epochs 10 --gs_lr 1.5e-3 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task jurisprudence \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

#! Submitted batch job 5044396 -> 219_courant -- mmlu_cross_task_epoch10

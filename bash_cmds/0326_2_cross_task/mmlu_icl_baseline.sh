# Off-diagonal ICL baselines for MMLU cross-task: source task demos, no CT-KV optimization
# makesbatch --time 4 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0326_2_cross_task/mmlu_icl_baseline.sh

# ct_mmlu_icl_cross_src_abstract_algebra_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_abstract_algebra_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task abstract_algebra \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_icl_cross_src_college_mathematics_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_college_mathematics_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task college_mathematics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_icl_cross_src_college_physics_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_college_physics_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task college_physics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_icl_cross_src_high_school_physics_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_high_school_physics_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task high_school_physics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_icl_cross_src_high_school_macroeconomics_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_high_school_macroeconomics_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task high_school_macroeconomics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_icl_cross_src_high_school_microeconomics_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_high_school_microeconomics_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task high_school_microeconomics \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_icl_cross_src_college_biology_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_college_biology_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task college_biology \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_icl_cross_src_high_school_biology_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_high_school_biology_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task high_school_biology \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_icl_cross_src_international_law_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_international_law_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task international_law \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ct_mmlu_icl_cross_src_jurisprudence_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_icl_cross_src_jurisprudence_seed42 \
    --seed 42 --eval_ratio 1.0 \
    --demo_source_task jurisprudence \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

#! Submitted batch job 5070956 -> 36_mren -- mmlu_icl_baseline

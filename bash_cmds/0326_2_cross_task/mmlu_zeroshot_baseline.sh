# Zero-shot baseline for MMLU cross-task: no demos, no optimization
# makesbatch --time 4 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0326_2_cross_task/mmlu_zeroshot_baseline.sh

# ct_mmlu_zeroshot_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag ct_mmlu_zeroshot_seed42 \
    --seed 42 --eval_ratio 1.0 --zero_shot \
    --select_tasks abstract_algebra college_mathematics college_physics high_school_physics high_school_macroeconomics high_school_microeconomics college_biology high_school_biology international_law jurisprudence

# ran locally
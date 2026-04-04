# Zero-shot baselines for BBH cross-task: no demos, no optimization
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0326_2_cross_task/bbh_zeroshot_baseline.sh

# ct_bbh_zeroshot_logdeduc_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_zeroshot_logdeduc_seed42 \
    --seed 42 --max_seq_len 4096 --zero_shot \
    --select_tasks logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects

# ct_bbh_zeroshot_tracking_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_zeroshot_tracking_seed42 \
    --seed 42 --max_seq_len 4096 --zero_shot \
    --select_tasks tracking_shuffled_objects_three_objects tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects

# ran locally
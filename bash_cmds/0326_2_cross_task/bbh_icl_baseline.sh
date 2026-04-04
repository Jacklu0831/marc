# Off-diagonal ICL baselines for BBH cross-task: source task demos, no CT-KV optimization
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0326_2_cross_task/bbh_icl_baseline.sh

# --- logical_deduction family ---

# ct_bbh_icl_cross_logdeduc_src_three_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_icl_cross_logdeduc_src_three_seed42 \
    --seed 42 --max_seq_len 4096 \
    --demo_source_task logical_deduction_three_objects \
    --select_tasks logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects

# ct_bbh_icl_cross_logdeduc_src_five_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_icl_cross_logdeduc_src_five_seed42 \
    --seed 42 --max_seq_len 4096 \
    --demo_source_task logical_deduction_five_objects \
    --select_tasks logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects

# ct_bbh_icl_cross_logdeduc_src_seven_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_icl_cross_logdeduc_src_seven_seed42 \
    --seed 42 --max_seq_len 4096 \
    --demo_source_task logical_deduction_seven_objects \
    --select_tasks logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects

# --- tracking_shuffled_objects family ---

# ct_bbh_icl_cross_tracking_src_three_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_icl_cross_tracking_src_three_seed42 \
    --seed 42 --max_seq_len 4096 \
    --demo_source_task tracking_shuffled_objects_three_objects \
    --select_tasks tracking_shuffled_objects_three_objects tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects

# ct_bbh_icl_cross_tracking_src_five_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_icl_cross_tracking_src_five_seed42 \
    --seed 42 --max_seq_len 4096 \
    --demo_source_task tracking_shuffled_objects_five_objects \
    --select_tasks tracking_shuffled_objects_three_objects tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects

# ct_bbh_icl_cross_tracking_src_seven_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_icl_cross_tracking_src_seven_seed42 \
    --seed 42 --max_seq_len 4096 \
    --demo_source_task tracking_shuffled_objects_seven_objects \
    --select_tasks tracking_shuffled_objects_three_objects tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects

#! Submitted batch job 5070948 -> 36_mren -- bbh_icl_baseline

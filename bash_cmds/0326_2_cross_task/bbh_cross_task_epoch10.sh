# Cross-task KV transfer on BBH: logical_deduction and tracking_shuffled_objects families
# Addresses R1 Q1: "whether optimized contexts can generalize across related tasks"
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0326_2_cross_task/bbh_cross_task_epoch10.sh

# --- logical_deduction family: 3x3 matrix ---

# ct_bbh_cross_logdeduc_src_three_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_cross_logdeduc_src_three_ep10_seed42 \
    --seed 42 --max_seq_len 4096 \
    --gs_epochs 10 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task logical_deduction_three_objects \
    --select_tasks logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects

# ct_bbh_cross_logdeduc_src_five_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_cross_logdeduc_src_five_ep10_seed42 \
    --seed 42 --max_seq_len 4096 \
    --gs_epochs 10 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task logical_deduction_five_objects \
    --select_tasks logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects

# ct_bbh_cross_logdeduc_src_seven_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_cross_logdeduc_src_seven_ep10_seed42 \
    --seed 42 --max_seq_len 4096 \
    --gs_epochs 10 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task logical_deduction_seven_objects \
    --select_tasks logical_deduction_three_objects logical_deduction_five_objects logical_deduction_seven_objects

# --- tracking_shuffled_objects family: 3x3 matrix ---

# ct_bbh_cross_tracking_src_three_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_cross_tracking_src_three_ep10_seed42 \
    --seed 42 --max_seq_len 4096 \
    --gs_epochs 10 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task tracking_shuffled_objects_three_objects \
    --select_tasks tracking_shuffled_objects_three_objects tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects

# ct_bbh_cross_tracking_src_five_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_cross_tracking_src_five_ep10_seed42 \
    --seed 42 --max_seq_len 4096 \
    --gs_epochs 10 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task tracking_shuffled_objects_five_objects \
    --select_tasks tracking_shuffled_objects_three_objects tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects

# ct_bbh_cross_tracking_src_seven_ep10_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag ct_bbh_cross_tracking_src_seven_ep10_seed42 \
    --seed 42 --max_seq_len 4096 \
    --gs_epochs 10 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1 \
    --demo_source_task tracking_shuffled_objects_seven_objects \
    --select_tasks tracking_shuffled_objects_three_objects tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects

#! Submitted batch job 5056040 -> 36_mren -- bbh_cross_task_epoch10

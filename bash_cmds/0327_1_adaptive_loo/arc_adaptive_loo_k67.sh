# Adaptive LOO K=6,7 sweep (extending arc_adaptive_loo.sh)
# Same config: lr=3e-3, epochs=200, tokdrop=0.1
# makesbatch --time 4 --ngpu 1 --gb 64 --l40s --bash_file bash_cmds/0327_1_adaptive_loo/arc_adaptive_loo_k67.sh

# --- Adaptive LOO K=6 (LOO for tasks with >=6 demos, 4.5% of tasks = 18 tasks) ---

# arc_adaptloo_k6_part1
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k6_part1 \
    --select_tasks_path data/task_info_part1.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 6

# arc_adaptloo_k6_part2
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k6_part2 \
    --select_tasks_path data/task_info_part2.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 6

# arc_adaptloo_k6_part3
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k6_part3 \
    --select_tasks_path data/task_info_part3.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 6

# arc_adaptloo_k6_part4
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k6_part4 \
    --select_tasks_path data/task_info_part4.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 6

# arc_adaptloo_k6_part5
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k6_part5 \
    --select_tasks_path data/task_info_part5.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 6

# --- Adaptive LOO K=7 (LOO for tasks with >=7 demos, 1% of tasks = 4 tasks) ---

# arc_adaptloo_k7_part1
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k7_part1 \
    --select_tasks_path data/task_info_part1.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 7

# arc_adaptloo_k7_part2
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k7_part2 \
    --select_tasks_path data/task_info_part2.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 7

# arc_adaptloo_k7_part3
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k7_part3 \
    --select_tasks_path data/task_info_part3.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 7

# arc_adaptloo_k7_part4
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k7_part4 \
    --select_tasks_path data/task_info_part4.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 7

# arc_adaptloo_k7_part5
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k7_part5 \
    --select_tasks_path data/task_info_part5.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 7

#! Submitted batch job 5031380 -> 36_mren -- arc_adaptloo_k6_part1
#! Submitted batch job 5031381 -> 219_courant -- arc_adaptloo_k6_part2
#! Submitted batch job 5031382 -> 36_general -- arc_adaptloo_k6_part3
#! Submitted batch job 5031383 -> 36_cds -- arc_adaptloo_k6_part4
#! Submitted batch job 5031384 -> 36_mren -- arc_adaptloo_k6_part5
#! Submitted batch job 5031385 -> 219_courant -- arc_adaptloo_k7_part1
#! Submitted batch job 5031386 -> 36_mren -- arc_adaptloo_k7_part2
#! Submitted batch job 5031387 -> 219_courant -- arc_adaptloo_k7_part3
#! Submitted batch job 5031388 -> 36_mren -- arc_adaptloo_k7_part4
#! Submitted batch job 5031389 -> 36_mren -- arc_adaptloo_k7_part5

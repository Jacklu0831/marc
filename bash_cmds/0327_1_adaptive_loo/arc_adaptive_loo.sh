# Adaptive LOO masking sweep on ARC: enable LOO when #demos >= K, else disable
# Ablation config matching Table 3 (lr=3e-3, epochs=200, tokdrop=0.1) + gs_loo_min_demos K sweep
# Baselines: no LOO (gs_dropout=none) and full LOO (gs_dropout=train)
# makesbatch --time 4 --ngpu 1 --gb 64 --l40s --bash_file bash_cmds/0327_1_adaptive_loo/arc_adaptive_loo.sh

# --- No LOO baseline (gs_dropout=none) ---

# arc_noloo_part1
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_noloo_part1 \
    --select_tasks_path data/task_info_part1.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout none --gs_token_dropout 0.1

# arc_noloo_part2
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_noloo_part2 \
    --select_tasks_path data/task_info_part2.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout none --gs_token_dropout 0.1

# arc_noloo_part3
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_noloo_part3 \
    --select_tasks_path data/task_info_part3.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout none --gs_token_dropout 0.1

# arc_noloo_part4
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_noloo_part4 \
    --select_tasks_path data/task_info_part4.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout none --gs_token_dropout 0.1

# arc_noloo_part5
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_noloo_part5 \
    --select_tasks_path data/task_info_part5.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout none --gs_token_dropout 0.1

# --- Full LOO (gs_dropout=train for all tasks) ---

# arc_fullloo_part1
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_fullloo_part1 \
    --select_tasks_path data/task_info_part1.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_fullloo_part2
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_fullloo_part2 \
    --select_tasks_path data/task_info_part2.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_fullloo_part3
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_fullloo_part3 \
    --select_tasks_path data/task_info_part3.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_fullloo_part4
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_fullloo_part4 \
    --select_tasks_path data/task_info_part4.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_fullloo_part5
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_fullloo_part5 \
    --select_tasks_path data/task_info_part5.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1

# --- Adaptive LOO K=3 (LOO for tasks with >=3 demos, 89% of tasks) ---

# arc_adaptloo_k3_part1
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k3_part1 \
    --select_tasks_path data/task_info_part1.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 3

# arc_adaptloo_k3_part2
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k3_part2 \
    --select_tasks_path data/task_info_part2.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 3

# arc_adaptloo_k3_part3
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k3_part3 \
    --select_tasks_path data/task_info_part3.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 3

# arc_adaptloo_k3_part4
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k3_part4 \
    --select_tasks_path data/task_info_part4.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 3

# arc_adaptloo_k3_part5
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k3_part5 \
    --select_tasks_path data/task_info_part5.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 3

# --- Adaptive LOO K=4 (LOO for tasks with >=4 demos, 35% of tasks) ---

# arc_adaptloo_k4_part1
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k4_part1 \
    --select_tasks_path data/task_info_part1.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 4

# arc_adaptloo_k4_part2
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k4_part2 \
    --select_tasks_path data/task_info_part2.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 4

# arc_adaptloo_k4_part3
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k4_part3 \
    --select_tasks_path data/task_info_part3.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 4

# arc_adaptloo_k4_part4
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k4_part4 \
    --select_tasks_path data/task_info_part4.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 4

# arc_adaptloo_k4_part5
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k4_part5 \
    --select_tasks_path data/task_info_part5.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 4

# --- Adaptive LOO K=5 (LOO for tasks with >=5 demos, 12% of tasks) ---

# arc_adaptloo_k5_part1
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k5_part1 \
    --select_tasks_path data/task_info_part1.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 5

# arc_adaptloo_k5_part2
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k5_part2 \
    --select_tasks_path data/task_info_part2.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 5

# arc_adaptloo_k5_part3
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k5_part3 \
    --select_tasks_path data/task_info_part3.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 5

# arc_adaptloo_k5_part4
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k5_part4 \
    --select_tasks_path data/task_info_part4.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 5

# arc_adaptloo_k5_part5
.venv/bin/accelerate launch --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --tag arc_adaptloo_k5_part5 \
    --select_tasks_path data/task_info_part5.csv --no_bos \
    --weight_dir 0317_noprogram_base --weight_epoch 24 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_token_dropout 0.1 --gs_loo_min_demos 5

#! Submitted batch job 5006899 -> 36_mren -- arc_noloo_part1
#! Submitted batch job 5006900 -> 219_courant -- arc_noloo_part2
#! Submitted batch job 5006901 -> 36_general -- arc_noloo_part3
#! Submitted batch job 5006902 -> 36_cds -- arc_noloo_part4
#! Submitted batch job 5006903 -> 36_mren -- arc_noloo_part5
#! Submitted batch job 5006904 -> 219_courant -- arc_fullloo_part1
#! Submitted batch job 5006905 -> 36_general -- arc_fullloo_part2
#! Submitted batch job 5006906 -> 36_cds -- arc_fullloo_part3
#! Submitted batch job 5006907 -> 36_mren -- arc_fullloo_part4
#! Submitted batch job 5006908 -> 219_courant -- arc_fullloo_part5
#! Submitted batch job 5006909 -> 36_mren -- arc_adaptloo_k3_part1
#! Submitted batch job 5006910 -> 219_courant -- arc_adaptloo_k3_part2
#! Submitted batch job 5006911 -> 36_mren -- arc_adaptloo_k3_part3
#! Submitted batch job 5006912 -> 219_courant -- arc_adaptloo_k3_part4
#! Submitted batch job 5006913 -> 36_mren -- arc_adaptloo_k3_part5
#! Submitted batch job 5006914 -> 219_courant -- arc_adaptloo_k4_part1
#! Submitted batch job 5006915 -> 36_mren -- arc_adaptloo_k4_part2
#! Submitted batch job 5006916 -> 219_courant -- arc_adaptloo_k4_part3
#! Submitted batch job 5006917 -> 36_mren -- arc_adaptloo_k4_part4
#! Submitted batch job 5006918 -> 219_courant -- arc_adaptloo_k4_part5
#! Submitted batch job 5006919 -> 36_mren -- arc_adaptloo_k5_part1
#! Submitted batch job 5006920 -> 219_courant -- arc_adaptloo_k5_part2
#! Submitted batch job 5006921 -> 36_mren -- arc_adaptloo_k5_part3
#! Submitted batch job 5006922 -> 219_courant -- arc_adaptloo_k5_part4
#! Submitted batch job 5006923 -> 36_mren -- arc_adaptloo_k5_part5

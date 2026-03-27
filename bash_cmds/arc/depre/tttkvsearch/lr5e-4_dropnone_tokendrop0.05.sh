# arc gs5 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 5 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs10 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs10_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 10 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs15 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs15_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 15 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs20 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs20_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 20 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs25 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 25 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs30 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs30_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 30 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs35 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs35_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 35 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs40 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs40_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 40 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs45 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs45_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 45 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs50 lr5e-4 dropnone tokendrop0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs50_lr5e-4_dropnone_tokendrop0.05 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 50 \
    --gs_lr 5e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# 0.3
# 0.3125
# 0.3125
# 0.3125
# 0.3
# 0.3
# 0.3125
# 0.3
# 0.3
# 0.3
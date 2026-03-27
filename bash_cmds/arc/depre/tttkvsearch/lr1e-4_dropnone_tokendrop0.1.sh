# arc gs5 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs10 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs10_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs15 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs15_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 15 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs20 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs20_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs25 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs30 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs30_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 30 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs35 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs35_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 35 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs40 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs40_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 40 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs45 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs45_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 45 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs50 lr1e-4 dropnone tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs50_lr1e-4_dropnone_tokendrop0.1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter250_save_seed0_0317_noprogram_base \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# 0.3
# 0.3
# 0.3
# 0.3125
# 0.3
# 0.3
# 0.3125
# 0.3125
# 0.3
# 0.3
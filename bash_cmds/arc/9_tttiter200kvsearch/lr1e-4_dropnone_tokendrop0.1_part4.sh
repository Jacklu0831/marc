# well under 2hrs

# arc ttt gs10 lr1e-4 dropnone tokendrop0.1 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_ttt_gs10_lr1e-4_dropnone_tokendrop0.1_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter200_save_seed0_part4_0317_noprogram_base \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc ttt gs20 lr1e-4 dropnone tokendrop0.1 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_ttt_gs20_lr1e-4_dropnone_tokendrop0.1_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter200_save_seed0_part4_0317_noprogram_base \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc ttt gs30 lr1e-4 dropnone tokendrop0.1 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_ttt_gs30_lr1e-4_dropnone_tokendrop0.1_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter200_save_seed0_part4_0317_noprogram_base \
    --gs_epochs 30 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc ttt gs40 lr1e-4 dropnone tokendrop0.1 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_ttt_gs40_lr1e-4_dropnone_tokendrop0.1_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter200_save_seed0_part4_0317_noprogram_base \
    --gs_epochs 40 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc ttt gs50 lr1e-4 dropnone tokendrop0.1 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_ttt_gs50_lr1e-4_dropnone_tokendrop0.1_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter200_save_seed0_part4_0317_noprogram_base \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# 0.2625
# 0.2625
# 0.2625
# 0.2625
# 0.2625
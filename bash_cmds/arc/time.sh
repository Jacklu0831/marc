# need to rescale all of these

# p-tuning 32token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arctime3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32

# p-tuning demotoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arctime4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token

# kv-tuning 32token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arctime5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --random_kv token \
    --random_kv_ntokens 32

# kv-tuning demotoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arctime6 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --random_kv token

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arctime7 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 50 \
    --ttt_permute_n 1000 \
    --seed 0

# ct-p
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arctime1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# ct-kv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arctime2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0

# tttkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arctime8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_ttt_iter200_save_seed1_part1_0317_noprogram_base \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# 4.013169471810504
# 17.506513929948575
# 3.6972249891699813
# 5.769035333540382
# 12.680275277393621
# 17.487284087553256
# 5.768550358167508
# 5.811798049182427
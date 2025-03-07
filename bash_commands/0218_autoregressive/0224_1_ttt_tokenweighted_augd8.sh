# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0218_autoregressive/0224_1_ttt_tokenweighted_augd8.sh --rtx8000





# # eval ar tokenweighted epoch0
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
#     --tag epoch0 \
#     --weight_dir 0218_ar_tokenweighted \
#     --weight_epoch 24 \
#     --select_tasks_path task_info_selected.csv

# # eval ar tokenweighted epoch0 alltask
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
#     --tag epoch0_alltask \
#     --weight_dir 0218_ar_tokenweighted \
#     --weight_epoch 24





# # ttt ar tokenweighted augd8
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/ttt.py \
#     --select_tasks_path task_info_selected.csv \
#     --weight_dir 0218_ar_tokenweighted \
#     --weight_epoch 24 \
#     --tag augd8 \
#     --num_epochs 5 \
#     --save_epochs 1 \
#     --aug_type d8



# eval ar tokenweighted augd8 epoch1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch1 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0218_ar_tokenweighted \
    --ttt_weight_epoch 1 \
    --select_tasks_path task_info_selected.csv

# eval ar tokenweighted augd8 epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch2 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0218_ar_tokenweighted \
    --ttt_weight_epoch 2 \
    --select_tasks_path task_info_selected.csv

# eval ar tokenweighted augd8 epoch3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch3 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0218_ar_tokenweighted \
    --ttt_weight_epoch 3 \
    --select_tasks_path task_info_selected.csv

# eval ar tokenweighted augd8 epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch4 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0218_ar_tokenweighted \
    --ttt_weight_epoch 4 \
    --select_tasks_path task_info_selected.csv

# eval ar tokenweighted augd8 epoch5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch5 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0218_ar_tokenweighted \
    --ttt_weight_epoch 5 \
    --select_tasks_path task_info_selected.csv




# EVALUATE BEST TTT WITH VOTING (can try different config for leave_ns)
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0218_ar_tokenweighted \
#     --weight_epoch 24 \
#     --ttt_weight_dir ttt_augd8_0218_ar_tokenweighted \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv
#     --select_tasks_path task_info_selected.csv
#     --leave_ns 1 \
#     --permute_n 2 \
#     --augment_n 5 \
#     --leave_ns_inc



# Submitted batch job 57709401 # eval on 80 tasks, worse than all tasks
# Submitted batch job 57709402 # eval on tasks, slightly higher than training, excuse me
# Submitted batch job 57709403 # not OOM! 28hr

# Submitted batch job 57852670
# Submitted batch job 57852671
# Submitted batch job 57852672
# Submitted batch job 57852673
# Submitted batch job 57852674
# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0218_autoregressive/0224_2_ttt_tokenweighted_augextra.sh --rtx8000

# # ttt ar tokenweighted augextra
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/ttt.py \
#     --select_tasks_path task_info_selected.csv \
#     --weight_dir 0218_ar_tokenweighted \
#     --weight_epoch 24 \
#     --tag augextra \
#     --num_epochs 5 \
#     --save_epochs 1 \
#     --aug_type extra



# eval ar tokenweighted augextra epoch1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch1 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0218_ar_tokenweighted \
    --ttt_weight_epoch 1 \
    --select_tasks_path task_info_selected.csv

# eval ar tokenweighted augextra epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch2 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0218_ar_tokenweighted \
    --ttt_weight_epoch 2 \
    --select_tasks_path task_info_selected.csv

# eval ar tokenweighted augextra epoch3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch3 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0218_ar_tokenweighted \
    --ttt_weight_epoch 3 \
    --select_tasks_path task_info_selected.csv

# eval ar tokenweighted augextra epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch4 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0218_ar_tokenweighted \
    --ttt_weight_epoch 4 \
    --select_tasks_path task_info_selected.csv

# eval ar tokenweighted augextra epoch5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
    --tag epoch5 \
    --weight_dir 0218_ar_tokenweighted \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0218_ar_tokenweighted \
    --ttt_weight_epoch 5 \
    --select_tasks_path task_info_selected.csv




# EVALUATE BEST TTT WITH VOTING (can try different config for leave_ns)
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0218_ar_tokenweighted \
#     --weight_epoch 24 \
#     --ttt_weight_dir ttt_augextra_0218_ar_tokenweighted \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv
#     --select_tasks_path task_info_selected.csv
#     --leave_ns 1 \
#     --permute_n 2 \
#     --augment_n 5 \
#     --leave_ns_inc



# Submitted batch job 57709404 # not OOM! 43 hr

# Submitted batch job 57852735
# Submitted batch job 57852736
# Submitted batch job 57852737
# Submitted batch job 57852738
# Submitted batch job 57852739
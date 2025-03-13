# for eval, run rtx8000
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0222_noprogram/0224_2_ttt_padleft_augd8.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 12 --bash_files bash_commands/0222_noprogram/0224_2_ttt_padleft_augd8.sh





# eval noprogram padleft epoch0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch0 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --select_tasks_path task_info_selected.csv

# eval noprogram padleft epoch0 alltask
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch0_alltask \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26





# # ttt noprogram padleft augd8
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
#     --lr_scheduler constant \
#     --select_tasks_path task_info_selected.csv \
#     --weight_dir 0224_noprogram_padleft \
#     --weight_epoch 26 \
#     --tag augd8 \
#     --num_epochs 5 \
#     --save_epochs 1 \
#     --aug_type d8




# eval noprogram padleft augd8 epoch1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch1 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augd8_0224_noprogram_padleft \
    --ttt_weight_epoch 1 \
    --select_tasks_path task_info_selected.csv

# eval noprogram padleft augd8 epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch2 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augd8_0224_noprogram_padleft \
    --ttt_weight_epoch 2 \
    --select_tasks_path task_info_selected.csv

# eval noprogram padleft augd8 epoch3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch3 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augd8_0224_noprogram_padleft \
    --ttt_weight_epoch 3 \
    --select_tasks_path task_info_selected.csv

# eval noprogram padleft augd8 epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch4 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augd8_0224_noprogram_padleft \
    --ttt_weight_epoch 4 \
    --select_tasks_path task_info_selected.csv

# eval noprogram padleft augd8 epoch5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch5 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augd8_0224_noprogram_padleft \
    --ttt_weight_epoch 5 \
    --select_tasks_path task_info_selected.csv




# EVALUATE BEST TTT WITH VOTING (can try different config for leave_ns)
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0224_noprogram_padleft \
#     --weight_epoch 26 \
#     --ttt_weight_dir ttt_augd8_0224_noprogram_padleft \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv
#     --select_tasks_path task_info_selected.csv
#     --leave_ns 1 \
#     --permute_n 2 \
#     --augment_n 5 \
#     --leave_ns_inc


# re-eval on rtx8000
# Submitted batch job 58104828 # 0.125, matches with before
# Submitted batch job 58104829 # 0.0875 # matches with before

# ttt (2hrs on a100)
# Submitted batch job 57918007

# eval ttt on rtx8000
# Submitted batch job 58104830 # 0.2375
# Submitted batch job 58104831 # 0.2625
# Submitted batch job 58104832 # 0.2625
# Submitted batch job 58104833 # 0.275
# Submitted batch job 58104834 # 0.2625
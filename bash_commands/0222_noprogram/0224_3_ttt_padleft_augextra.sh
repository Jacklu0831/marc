# for eval, run rtx8000
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0222_noprogram/0224_3_ttt_padleft_augextra.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 12 --bash_files bash_commands/0222_noprogram/0224_3_ttt_padleft_augextra.sh



# # ttt noprogram padleft augextra
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
#     --lr_scheduler constant \
#     --select_tasks_path task_info_selected.csv \
#     --weight_dir 0224_noprogram_padleft \
#     --weight_epoch 26 \
#     --tag augextra \
#     --num_epochs 5 \
#     --save_epochs 1 \
#     --aug_type extra





# eval noprogram padleft extra epoch1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch1 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augextra_0224_noprogram_padleft \
    --ttt_weight_epoch 1 \
    --select_tasks_path task_info_selected.csv

# eval noprogram padleft extra epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch2 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augextra_0224_noprogram_padleft \
    --ttt_weight_epoch 2 \
    --select_tasks_path task_info_selected.csv

# eval noprogram padleft extra epoch3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch3 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augextra_0224_noprogram_padleft \
    --ttt_weight_epoch 3 \
    --select_tasks_path task_info_selected.csv

# eval noprogram padleft extra epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch4 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augextra_0224_noprogram_padleft \
    --ttt_weight_epoch 4 \
    --select_tasks_path task_info_selected.csv

# eval noprogram padleft extra epoch5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch5 \
    --weight_dir 0224_noprogram_padleft \
    --weight_epoch 26 \
    --ttt_weight_dir ttt_augextra_0224_noprogram_padleft \
    --ttt_weight_epoch 5 \
    --select_tasks_path task_info_selected.csv




# EVALUATE BEST TTT WITH VOTING (can try different config for leave_ns)
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0224_noprogram_padleft \
#     --weight_epoch 26 \
#     --ttt_weight_dir ttt_augextra_0224_noprogram_padleft \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv
#     --select_tasks_path task_info_selected.csv
#     --leave_ns 1 \
#     --permute_n 2 \
#     --augment_n 5 \
#     --leave_ns_inc


# ttt
# Submitted batch job 57918010

# eval ttt on rtx8000
# Submitted batch job 58107162 # 0.2375
# Submitted batch job 58107163 # 0.225
# Submitted batch job 58107164 # 0.2
# Submitted batch job 58107165 # 0.1875
# Submitted batch job 58107166 # 0.2175
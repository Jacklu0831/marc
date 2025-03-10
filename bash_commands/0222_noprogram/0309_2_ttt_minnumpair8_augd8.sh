# for eval, run rtx8000
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0222_noprogram/0309_2_ttt_minnumpair8_augd8.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 12 --bash_files bash_commands/0222_noprogram/0309_2_ttt_minnumpair8_augd8.sh



# # eval noprogram minnumpair8 epoch0
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --tag epoch0 \
#     --weight_dir 0222_noprogram_base_minnumpair8 \
#     --weight_epoch 22 \
#     --select_tasks_path task_info_selected.csv

# # eval noprogram minnumpair8 epoch0 alltask
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --tag epoch0_alltask \
#     --weight_dir 0222_noprogram_base_minnumpair8 \
#     --weight_epoch 22



# # ttt noprogram minnumpair8 augd8
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
#     --lr_scheduler constant \
#     --select_tasks_path task_info_selected.csv \
#     --weight_dir 0222_noprogram_base_minnumpair8 \
#     --weight_epoch 22 \
#     --tag augd8 \
#     --num_epochs 10 \
#     --save_epochs 2 \
#     --aug_type d8





# eval noprogram minnumpair8 d8 epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch2 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augd8_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 2 \
    --select_tasks_path task_info_selected.csv

# eval noprogram minnumpair8 d8 epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch4 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augd8_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 4 \
    --select_tasks_path task_info_selected.csv

# eval noprogram minnumpair8 d8 epoch6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch6 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augd8_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 6 \
    --select_tasks_path task_info_selected.csv

# eval noprogram minnumpair8 d8 epoch8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch8 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augd8_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 8 \
    --select_tasks_path task_info_selected.csv

# eval noprogram minnumpair8 d8 epoch10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch10 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augd8_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 10 \
    --select_tasks_path task_info_selected.csv




# eval original
# Submitted batch job 58107220 # 0.1625
# Submitted batch job 58107221 # 0.115 (1/400 more task solved than before???)

# ttt
# Submitted batch job 58114904 # 4.4hr

# eval ttt
# Submitted batch job 58139571
# Submitted batch job 58139572
# Submitted batch job 58139573
# Submitted batch job 58139574
# Submitted batch job 58139575
# for eval, run rtx8000
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0222_noprogram/0309_2_ttt_minnumpair8_augextra.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 12 --bash_files bash_commands/0222_noprogram/0309_2_ttt_minnumpair8_augextra.sh




# # ttt noprogram minnumpair8 augextra
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
#     --lr_scheduler constant \
#     --select_tasks_path task_info_selected.csv \
#     --weight_dir 0222_noprogram_base_minnumpair8 \
#     --weight_epoch 22 \
#     --tag augextra \
#     --num_epochs 10 \
#     --save_epochs 2 \
#     --aug_type extra





# eval noprogram minnumpair8 extra epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch2 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augextra_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 2 \
    --select_tasks_path task_info_selected.csv

# eval noprogram minnumpair8 extra epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch4 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augextra_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 4 \
    --select_tasks_path task_info_selected.csv

# eval noprogram minnumpair8 extra epoch6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch6 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augextra_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 6 \
    --select_tasks_path task_info_selected.csv

# eval noprogram minnumpair8 extra epoch8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch8 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augextra_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 8 \
    --select_tasks_path task_info_selected.csv

# eval noprogram minnumpair8 extra epoch10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag epoch10 \
    --weight_dir 0222_noprogram_base_minnumpair8 \
    --weight_epoch 22 \
    --ttt_weight_dir ttt_augextra_0222_noprogram_base_minnumpair8 \
    --ttt_weight_epoch 10 \
    --select_tasks_path task_info_selected.csv




# ttt
# Submitted batch job 58114901 # 5hr

# eval ttt
# Submitted batch job 58139577
# Submitted batch job 58139578
# Submitted batch job 58139579
# Submitted batch job 58139580
# Submitted batch job 58139581
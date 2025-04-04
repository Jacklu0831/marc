# for eval, run rtx8000
# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0222_noprogram/0224_1_ttt_noft.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0222_noprogram/0224_1_ttt_noft.sh





# time: 4.2 hrs on single a100
# disk memory: 293M for lora * 5 epoch * 80 task = 118GB (in-reality 115GB)

# # run locally
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/train.py \
#     --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
#     --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
#     --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
#     --lr_scheduler constant \
#     --tag 0224_noprogram_noft \
#     --lr_embedding 0.0 \
#     --lr_program 0.0 \
#     --lr_prior 0.0 \
#     --lr_other 0.0 \
#     --samples_per_epoch 16 \
#     --num_epochs 2


# # eval noprogram noft augextra epoch0
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/evaluate.py \
#     --tag epoch0 \
#     --weight_dir 0224_noprogram_noft \
#     --weight_epoch 2 \
#     --select_tasks_path task_info_selected.csv

# # eval noprogram noft augextra epoch0 alltask
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/evaluate.py \
#     --tag epoch0_alltask \
#     --weight_dir 0224_noprogram_noft \
#     --weight_epoch 2




# ttt noprogram noft augextra
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/ttt.py \
    --lr_scheduler constant \
    --select_tasks_path task_info_selected.csv \
    --weight_dir 0224_noprogram_noft \
    --weight_epoch 2 \
    --tag augextra \
    --num_epochs 5 \
    --save_epochs 1 \
    --aug_type extra




# # eval noprogram noft augextra epoch1
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/evaluate.py \
#     --tag epoch1 \
#     --weight_dir 0224_noprogram_noft \
#     --weight_epoch 2 \
#     --ttt_weight_dir ttt_augextra_0224_noprogram_noft \
#     --ttt_weight_epoch 1 \
#     --select_tasks_path task_info_selected.csv

# # eval noprogram noft augextra epoch2
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/evaluate.py \
#     --tag epoch2 \
#     --weight_dir 0224_noprogram_noft \
#     --weight_epoch 2 \
#     --ttt_weight_dir ttt_augextra_0224_noprogram_noft \
#     --ttt_weight_epoch 2 \
#     --select_tasks_path task_info_selected.csv

# # eval noprogram noft augextra epoch3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/evaluate.py \
#     --tag epoch3 \
#     --weight_dir 0224_noprogram_noft \
#     --weight_epoch 2 \
#     --ttt_weight_dir ttt_augextra_0224_noprogram_noft \
#     --ttt_weight_epoch 3 \
#     --select_tasks_path task_info_selected.csv

# # eval noprogram noft augextra epoch4
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/evaluate.py \
#     --tag epoch4 \
#     --weight_dir 0224_noprogram_noft \
#     --weight_epoch 2 \
#     --ttt_weight_dir ttt_augextra_0224_noprogram_noft \
#     --ttt_weight_epoch 4 \
#     --select_tasks_path task_info_selected.csv

# # eval noprogram noft augextra epoch5
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0224_noprogram_noft \
#     --weight_epoch 2 \
#     --ttt_weight_dir ttt_augextra_0224_noprogram_noft \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv




# EVALUATE BEST TTT WITH VOTING (can try different config for leave_ns)
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0224_noprogram_noft \
#     --weight_epoch 2 \
#     --ttt_weight_dir ttt_augextra_0224_noprogram_noft \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv
#     --leave_ns 1 \
#     --permute_n 2 \
#     --augment_n 5 \
#     --leave_ns_inc



# Submitted batch job 57709345 # got 0, expected
# Submitted batch job 57709346 # got 0, expected
# Submitted batch job 57709347 # OOM
# Submitted batch job 57917991 # rerun ttt
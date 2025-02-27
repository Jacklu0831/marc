# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0222_noprogram/0224_1_ttt_noft.sh
# time: <4 hrs on single a100
# disk memory: 293M for lora * 5 epoch * 80 task = 118GB

# # run locally
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
#     --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
#     --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
#     --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
#     --lr_scheduler constant \
#     --tag 0224_noprogram_no_ft \
#     --lr_embedding 0.0 \
#     --lr_program 0.0 \
#     --lr_prior 0.0 \
#     --lr_other 0.0 \
#     --samples_per_epoch 16 \
#     --num_epochs 2

# ttt noprogram base noft
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --select_tasks_path task_info_selected.csv \
    --weight_dir 0224_noprogram_no_ft \
    --weight_epoch 2 \
    --tag base \
    --num_epochs 5 \
    --save_epochs 1

# TOOD: add eval scripts here


# Submitted batch job 57602007 # ttt
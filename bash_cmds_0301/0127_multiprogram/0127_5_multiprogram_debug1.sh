# python make_sbatch.py --ngpu 2 --time 24 --bash_files bash_commands/0127_multiprogram/0127_5_multiprogram_debug1.sh

# overfit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_overfit \
    --only_train_original \
    --no_basic_aug \
    --samples_per_epoch 2500 \
    --num_epochs 100 \
    --wandb

# overfitfixedorder
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_overfitfixedorder \
    --only_train_original \
    --no_basic_aug \
    --samples_per_epoch 2500 \
    --num_epochs 100 \
    --debug_fixed_order \
    --wandb

# Submitted batch job 56639048
# Submitted batch job 56639049
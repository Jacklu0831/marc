# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0207_baseline/0207_0_base.sh

# noprogram base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --tag 0207_noprogram_base \
    --wandb

# noprogram voting
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0207_noprogram_voting \
    --wandb

# noprogram epoch35
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --tag 0207_noprogram_epoch35 \
    --num_epochs 35 \
    --wandb

# Submitted batch job 57063549
# Submitted batch job 57063551
# Submitted batch job 57063553
# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_noprogram/0211_0_base.sh

# noprogram base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0211_noprogram_base \
    --wandb

# noprogram nobasicaug
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0211_noprogram_nobasicaug \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --wandb

# # noprogram voting
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
#     --eval_eval_leave_ns 0 \
#     --eval_eval_permute_n 2 \
#     --eval_eval_augment_n 5 \
#     --eval_eval_leave_ns_inc \
#     --eval_eval_select_tasks_path task_info_selected.csv \
#     --tag 0211_noprogram_voting \
#     --wandb

# Submitted batch job 57149464
# Submitted batch job 57149465
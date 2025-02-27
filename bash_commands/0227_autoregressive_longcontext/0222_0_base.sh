# python make_sbatch.py --ngpu 2 --time 96 --bash_files bash_commands/0222_autoregressive_longcontext/0222_0_base.sh

# arlong
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0226/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0222_arlong \
    --long_context \
    --eval_batch_size 8 \
    --ar_gradient_checkpointing \
    --wandb

# arlong repeatdemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0226/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0222_arlong_repeatdemon \
    --long_context \
    --eval_batch_size 8 \
    --ar_gradient_checkpointing \
    --long_context_repeat_demonstration \
    --wandb

# # arlong repeatdemon lr4e-4
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0226/train.py \
#     --lr_scheduler constant \
#     --token_weighted_loss \
#     --tag 0222_arlong_repeatdemon_lr4e-4 \
#     --long_context \
#     --eval_batch_size 8 \
#     --ar_gradient_checkpointing \
#     --long_context_repeat_demonstration \
#     --lr_embedding 4e-5 \
#     --lr_program 4e-4 \
#     --lr_prior 4e-4 \
#     --lr_other 4e-4 \
#     --wandb

# Submitted batch job 57672421
# Submitted batch job 57672430
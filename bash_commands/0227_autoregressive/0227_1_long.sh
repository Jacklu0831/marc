# python make_sbatch.py --ngpu 2 --time 96 --bash_files bash_commands/0227_autoregressive/0227_1_long.sh

# arlong
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlong \
    --long_context \
    --train_batch_size 4 \
    --grad_accum_steps 4 \
    --eval_batch_size 8 \
    --gradient_checkpointing \
    --wandb

# arlong repeatdemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlong_repeatdemon \
    --long_context \
    --train_batch_size 4 \
    --grad_accum_steps 4 \
    --eval_batch_size 8 \
    --gradient_checkpointing \
    --long_context_repeat_demonstration \
    --wandb

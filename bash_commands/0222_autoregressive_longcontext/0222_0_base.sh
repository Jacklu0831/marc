# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0222_autoregressive_longcontext/0222_0_base.sh --multi_node

# arlong
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0222/train.py \
    --lr_scheduler constant \
    --tag 0222_arlong \
    --long_context \
    --train_batch_size 2 \
    --eval_batch_size 8 \
    --grad_accum_steps 4 \
    --checkpointing_threshold 3072 \
    --ar_gradient_checkpointing \
    --wandb

# arlong repeatdemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0222/train.py \
    --lr_scheduler constant \
    --tag 0222_arlong_repeatdemon \
    --long_context \
    --train_batch_size 2 \
    --eval_batch_size 8 \
    --grad_accum_steps 4 \
    --checkpointing_threshold 3072 \
    --ar_gradient_checkpointing \
    --long_context_repeat_demonstration \
    --wandb

# arlong repeatdemon lr4e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0222/train.py \
    --lr_scheduler constant \
    --tag 0222_arlong_repeatdemon_lr4e-4 \
    --long_context \
    --train_batch_size 2 \
    --eval_batch_size 8 \
    --grad_accum_steps 4 \
    --checkpointing_threshold 3072 \
    --ar_gradient_checkpointing \
    --long_context_repeat_demonstration \
    --lr_embedding 4e-5 \
    --lr_program 4e-4 \
    --lr_prior 4e-4 \
    --lr_other 4e-4 \
    --wandb

# Submitted batch job 57532048 # mengye
# Submitted batch job 57532049 # mengye
# Submitted batch job 57532050 # zhenbang
# python make_sbatch.py --ngpu 4 --time 168 --bash_files bash_commands/0227_autoregressive/0227_6_longcache_8b.sh
# python make_sbatch.py --ngpu 4 --time 168 --bash_files bash_commands/0227_autoregressive/0227_6_longcache_8b.sh --multi_node

# arlongcache llama8b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_llama8b \
    --model_name llama8b \
    --ar_gradient_checkpointing \
    --grad_accum_steps 4 \
    --untrainable_nbit 4 \
    --eval_batch_size 2 \
    --wandb


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag test \
    --model_name llama8b \
    --ar_gradient_checkpointing \
    --untrainable_nbit 4 \
    --grad_accum_steps 8 \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --debug_len 1867 \
    --log_every 1 \
    --samples_per_epoch 16

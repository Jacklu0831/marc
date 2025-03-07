# python make_sbatch.py --ngpu 4 --time 96 --bash_files bash_commands/0227_autoregressive/0227_6_longcache_3b.sh
# python make_sbatch.py --ngpu 4 --time 96 --bash_files bash_commands/0227_autoregressive/0227_6_longcache_3b.sh --multi_node
# thank god ar-gradient-checkpointing-only works
# halve grad accum step because 4 gpus now
# 4 days might not be enough but whatever

# arlongcache llama3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_llama3b \
    --model_name llama3b \
    --ar_gradient_checkpointing \
    --grad_accum_steps 4 \
    --untrainable_nbit 4 \
    --eval_batch_size 2 \
    --wandb

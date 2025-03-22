# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0322_autoregressive_nlp/0322_0_base.sh

# arlongcache nlp llama1b lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --eval_pretrained \
    --tag 0322_arlongcache_nlp_llama1b_lora \
    --wandb

# Submitted batch job 58652023
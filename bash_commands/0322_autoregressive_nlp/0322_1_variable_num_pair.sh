# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0322_autoregressive_nlp/0322_1_variable_num_pair.sh

# arlongcache nlp llama1b lora excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --eval_pretrained \
    --tag 0322_arlongcache_nlp_llama1b_lora_excludefirst \
    --loss_type exclude_first \
    --wandb

# arlongcache nlp llama1b lora evalminpair2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --eval_pretrained \
    --tag 0322_arlongcache_nlp_llama1b_lora_evalminpair2 \
    --eval_min_num_pair 2 \
    --wandb

# arlongcache nlp llama1b lora excludefirst evalminpair2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --eval_pretrained \
    --tag 0322_arlongcache_nlp_llama1b_lora_excludefirst_evalminpair2 \
    --loss_type exclude_first \
    --eval_min_num_pair 2 \
    --wandb

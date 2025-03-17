# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_autoregressive/0317_0_base.sh
# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0317_autoregressive/0317_0_base.sh

# arlongcache base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_base \
    --wandb

# arlongcache repro
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_repro \
    --wandb

# arlongcache nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_nolora \
    --no_lora \
    --wandb

# arlongcache minnumpair3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --min_num_pair 3 \
    --tag 0317_arlongcache_minnumpair3 \
    --wandb

# arlongcache 100task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_100task \
    --train_data_dir ./data/re-arc/train_data_100/tasks \
    --wandb

# arlongcache 200task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_200task \
    --train_data_dir ./data/re-arc/train_data_200/tasks \
    --wandb

# arlongcache noaug
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_noaug \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --wandb

# arlongcache extraaug0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_extraaug0.3 \
    --extra_augment_ratio 0.3 \
    --wandb

# arlongcache extraaug0.3single
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_extraaug0.3single \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --wandb

# arlongcache nomaxseqlen
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_nomaxseqlen \
    --max_seq_len 10000000 \
    --wandb

# arlongcache tokenweightedloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_arlongcache_tokenweightedloss \
    --wandb

# arlongcache llama3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_llama3b \
    --model_name llama3b \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --untrainable_nbit 4 \
    --wandb

# Submitted batch job 58466758
# Submitted batch job 58466759
# Submitted batch job 58466760
# Submitted batch job 58466761
# Submitted batch job 58466762
# Submitted batch job 58466763
# Submitted batch job 58466764
# Submitted batch job 58466765
# Submitted batch job 58466766
# Submitted batch job 58466767
# Submitted batch job 58466768
# Submitted batch job 58466771
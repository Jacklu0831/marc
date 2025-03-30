# python make_sbatch.py --ngpu 2 --time 48 --gb 32 --bash_files bash_commands/0317_autoregressive/0325_0_truncate.sh

# before launching, NEED TO experiment with increasing batch size, profile run time and adjust samples_per_epoch
# make sure memory works before launching, get a gpu with 32gb and one with 64gb, do not use cache, just normal training
#     with samples_per_epoch set low to test eval too but allocate appropriate dummy max cache before training
#     make_sbatch.py automatically multiplies gb with ngpu, so should be fine to just test on single gpu
# try overfitting before launching this too?

# arlongcache truncate
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_truncate \
    --short_context \
    --loss_on_first \
    --cache_size_per_task 50 \
    --prior_embed_ratio 0.1 \
    --no_color_permute \
    --min_num_pair 2 \
    --max_num_pair 2 \
    --samples_per_epoch 40000 \
    --train_batch_size 4 \
    --wandb

# arlongcache truncate priorratio0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_truncate_priorratio0.2 \
    --short_context \
    --loss_on_first \
    --cache_size_per_task 50 \
    --prior_embed_ratio 0.2 \
    --no_color_permute \
    --min_num_pair 2 \
    --max_num_pair 2 \
    --samples_per_epoch 40000 \
    --train_batch_size 4 \
    --wandb

# arlongcache truncate cachesize100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_truncate_cachesize100 \
    --short_context \
    --loss_on_first \
    --cache_size_per_task 100 \
    --prior_embed_ratio 0.1 \
    --no_color_permute \
    --min_num_pair 2 \
    --max_num_pair 2 \
    --samples_per_epoch 40000 \
    --train_batch_size 4 \
    --wandb

# arlongcache truncate nolossonfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_truncate_nolossonfirst \
    --short_context \
    --cache_size_per_task 50 \
    --prior_embed_ratio 0.1 \
    --no_color_permute \
    --min_num_pair 2 \
    --max_num_pair 2 \
    --samples_per_epoch 40000 \
    --train_batch_size 4 \
    --wandb

# Submitted batch job 58849258
# Submitted batch job 58849259
# Submitted batch job 58849260
# Submitted batch job 58849261
# for gs0, 1, 5, 25
# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_commands/0317_autoregressive/0324_0_gs.sh

# for more
# python make_sbatch.py --ngpu 1 --time 8 --bash_files bash_commands/0317_autoregressive/0324_0_gs.sh

# gs takes a bit longer than noprogram


# arlongcache eval gs0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs0 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24

# arlongcache eval gs1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs1 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 1 \
    --gs_take_best

# arlongcache eval gs5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs5 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 5 \
    --gs_take_best

# arlongcache eval gs25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs25 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 25 \
    --gs_take_best

# arlongcache eval gs100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs100 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 100 \
    --gs_take_best

# arlongcache eval gs200
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs200 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 200 \
    --gs_take_best








# arlongcache eval gs1 trainpastkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs1_trainpastkv \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 1 \
    --gs_take_best \
    --gs_train_past_kv

# arlongcache eval gs5 trainpastkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs5_trainpastkv \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 5 \
    --gs_take_best \
    --gs_train_past_kv

# arlongcache eval gs25 trainpastkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs25_trainpastkv \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 25 \
    --gs_take_best \
    --gs_train_past_kv

# arlongcache eval gs100 trainpastkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs100_trainpastkv \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 100 \
    --gs_take_best \
    --gs_train_past_kv

# arlongcache eval gs200 trainpastkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs200_trainpastkv \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 200 \
    --gs_take_best \
    --gs_train_past_kv


# no train past kv
# Submitted batch job 58706211
# Submitted batch job 58706212
# Submitted batch job 58706213
# Submitted batch job 58706214
# Submitted batch job 58706218
# Submitted batch job 58706219

# train past kv
# Submitted batch job 58706215
# Submitted batch job 58706216
# Submitted batch job 58706217
# Submitted batch job 58706220
# Submitted batch job 58706221
# for gs0, 1, 5, 25
# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_commands/0317_autoregressive/0324_0_gs.sh

# for more
# python make_sbatch.py --ngpu 1 --time 8 --bash_files bash_commands/0317_autoregressive/0324_0_gs.sh

# gs takes a bit longer than noprogram


# # arlongcache eval gs0
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs0 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24







# # arlongcache eval gs1
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs1 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 1 \
#     --gs_take_best

# # arlongcache eval gs5
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs5 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 5 \
#     --gs_take_best

# # arlongcache eval gs25
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs25 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 25 \
#     --gs_take_best

# # arlongcache eval gs100
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs100 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 100 \
#     --gs_take_best

# # arlongcache eval gs200
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs200 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 200 \
#     --gs_take_best


# gs0
# Submitted batch job 58720977 # 14.25 (exact same as original)

# no train past kv
# Submitted batch job 58720978 # 12.75
# Submitted batch job 58720979 # 16.25
# Submitted batch job 58720980 # 18
# Submitted batch job 58706218 # 19.5
# Submitted batch job 58706219 # 19.5








# # arlongcache eval gs1 trainpastkv lr1e-3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs1_trainpastkv_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 1 \
#     --gs_take_best \
#     --gs_train_past_kv \
#     --gs_lr 1e-3

# # arlongcache eval gs5 trainpastkv lr1e-3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs5_trainpastkv_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 5 \
#     --gs_take_best \
#     --gs_train_past_kv \
#     --gs_lr 1e-3

# # arlongcache eval gs25 trainpastkv lr1e-3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs25_trainpastkv_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 25 \
#     --gs_take_best \
#     --gs_train_past_kv \
#     --gs_lr 1e-3

# # arlongcache eval gs100 trainpastkv lr1e-3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs100_trainpastkv_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 100 \
#     --gs_take_best \
#     --gs_train_past_kv \
#     --gs_lr 1e-3

# # arlongcache eval gs200 trainpastkv lr1e-3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs200_trainpastkv_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 200 \
#     --gs_take_best \
#     --gs_train_past_kv \
#     --gs_lr 1e-3


# lower to 1e-3
# Submitted batch job 58758752 # 0.1475
# Submitted batch job 58758753 # 0.1575
# Submitted batch job 58758754 # 0.2
# Submitted batch job 58758759 # 0.2
# Submitted batch job 58758760 # 0.215







# # arlongcache eval gs1 trainpastkv lr1e-2
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs1_trainpastkv_lr1e-2 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 1 \
#     --gs_take_best \
#     --gs_train_past_kv \
#     --gs_lr 1e-2

# # arlongcache eval gs5 trainpastkv lr1e-2
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs5_trainpastkv_lr1e-2 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 5 \
#     --gs_take_best \
#     --gs_train_past_kv \
#     --gs_lr 1e-2

# # arlongcache eval gs25 trainpastkv lr1e-2
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
#     --no_bos \
#     --tag gs25_trainpastkv_lr1e-2 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_arlongcache_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 25 \
#     --gs_take_best \
#     --gs_train_past_kv \
#     --gs_lr 1e-2

# arlongcache eval gs100 trainpastkv lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs100_trainpastkv_lr1e-2 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 100 \
    --gs_take_best \
    --gs_train_past_kv \
    --gs_lr 1e-2

# arlongcache eval gs200 trainpastkv lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --no_bos \
    --tag gs200_trainpastkv_lr1e-2 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 200 \
    --gs_take_best \
    --gs_train_past_kv \
    --gs_lr 1e-2


# lr1e-2
# Submitted batch job 58790450
# Submitted batch job 58790451
# Submitted batch job 58790452
# Submitted batch job 58790453
# Submitted batch job 58790454
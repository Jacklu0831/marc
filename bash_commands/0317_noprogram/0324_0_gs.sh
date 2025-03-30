# for gs0, 1, 5, 25
# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_commands/0317_noprogram/0324_0_gs.sh

# for more
# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_commands/0317_noprogram/0324_0_gs.sh

# approx: for 400 tasks, gs0 takes ~16mins, gs10 takes ~20mins, gs50 takes ~60mins


# # noprogram eval gs0
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --no_bos \
#     --tag gs0 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_base \
#     --weight_epoch 24

# Submitted batch job 58706195 # 12.75 (same as before during training)








# # noprogram eval gs1
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --no_bos \
#     --tag gs1_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 1 \
#     --gs_take_best \
#     --gs_lr 1e-3

# # noprogram eval gs5
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --no_bos \
#     --tag gs5_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 5 \
#     --gs_take_best \
#     --gs_lr 1e-3

# # noprogram eval gs25
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --no_bos \
#     --tag gs25_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 25 \
#     --gs_take_best \
#     --gs_lr 1e-3

# # noprogram eval gs100
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --no_bos \
#     --tag gs100_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 100 \
#     --gs_take_best \
#     --gs_lr 1e-3

# # noprogram eval gs200
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --no_bos \
#     --tag gs200_lr1e-3 \
#     --batch_size 16 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_base \
#     --weight_epoch 24 \
#     --gs_batch_size 100000 \
#     --gs_iters 200 \
#     --gs_take_best \
#     --gs_lr 1e-3



# lower learning rate all the way to 1e-3
# Submitted batch job 58758676 # 0.1425
# Submitted batch job 58758677 # 0.1475
# Submitted batch job 58758678 # 0.175
# Submitted batch job 58758680 # 0.2
# Submitted batch job 58758681 # 0.205














# noprogram eval gs1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs1_lr1e-2 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 1 \
    --gs_take_best \
    --gs_lr 1e-2

# noprogram eval gs5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs5_lr1e-2 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 5 \
    --gs_take_best \
    --gs_lr 1e-2

# noprogram eval gs25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs25_lr1e-2 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 25 \
    --gs_take_best \
    --gs_lr 1e-2

# noprogram eval gs100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs100_lr1e-2 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 100 \
    --gs_take_best \
    --gs_lr 1e-2

# noprogram eval gs200
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs200_lr1e-2 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 200 \
    --gs_take_best \
    --gs_lr 1e-2


# lr1e-2
# Submitted batch job 58790438 # 0.1525
# Submitted batch job 58790439 # 0.1875
# Submitted batch job 58790440 # 0.1725
# Submitted batch job 58790441 # 0.15
# Submitted batch job 58790442 # 0.155
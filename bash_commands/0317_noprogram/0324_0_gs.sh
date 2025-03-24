# for gs0, 1, 5, 25
# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_commands/0317_noprogram/0324_0_gs.sh

# for more
# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_commands/0317_noprogram/0324_0_gs.sh

# approx: for 400 tasks, gs0 takes ~16mins, gs10 takes ~20mins, gs50 takes ~60mins


# noprogram eval gs0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs0 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24

# noprogram eval gs1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs1 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 1 \
    --gs_take_best

# noprogram eval gs5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs5 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 5 \
    --gs_take_best

# noprogram eval gs25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs25 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 25 \
    --gs_take_best

# noprogram eval gs100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs100 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 100 \
    --gs_take_best

# noprogram eval gs200
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag gs200 \
    --batch_size 16 \
    --flash_attn \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_batch_size 100000 \
    --gs_iters 200 \
    --gs_take_best


# Submitted batch job 58706195
# Submitted batch job 58706196
# Submitted batch job 58706197
# Submitted batch job 58706198
# Submitted batch job 58706201
# Submitted batch job 58706202
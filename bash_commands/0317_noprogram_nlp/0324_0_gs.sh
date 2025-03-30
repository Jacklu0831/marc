# for gs1, 5
# python make_sbatch.py --ngpu 1 --time 3 --bash_files bash_commands/0317_noprogram_nlp/0324_0_gs.sh

# for gs25
# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_commands/0317_noprogram_nlp/0324_0_gs.sh

# for gs100
# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0317_noprogram_nlp/0324_0_gs.sh

# approx: for eval ratio 0.1, gs0 takes literally ~3mins, gs25 takes ~5hrs due to very inefficient implementation
# to start making it more efficient: for each batch, if two demonstration pairs are the same, do not recompute gradient search
# if need more efficiency: caching kv


# # noprogram nlp eval gs0
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
#     --tag gs0 \
#     --batch_size 4 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_nlp_llama1b_lora \
#     --weight_epoch 6 \
#     --eval_ratio 0.1

# # noprogram nlp eval gs1 lr1e-3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
#     --tag gs1_lr1e-3 \
#     --batch_size 4 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_nlp_llama1b_lora \
#     --weight_epoch 6 \
#     --gs_batch_size 100000 \
#     --gs_iters 1 \
#     --gs_take_best \
#     --eval_ratio 0.03 \
#     --gs_lr 1e-3

# # noprogram nlp eval gs5 lr1e-3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
#     --tag gs5_lr1e-3 \
#     --batch_size 4 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_nlp_llama1b_lora \
#     --weight_epoch 6 \
#     --gs_batch_size 100000 \
#     --gs_iters 5 \
#     --gs_take_best \
#     --eval_ratio 0.1 \
#     --gs_lr 1e-3

# # noprogram nlp eval gs25 lr1e-3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
#     --tag gs25_lr1e-3 \
#     --batch_size 4 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_nlp_llama1b_lora \
#     --weight_epoch 6 \
#     --gs_batch_size 100000 \
#     --gs_iters 25 \
#     --gs_take_best \
#     --eval_ratio 0.1 \
#     --gs_lr 1e-3

# # noprogram nlp eval gs100 lr1e-3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
#     --tag gs100_lr1e-3 \
#     --batch_size 4 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_nlp_llama1b_lora \
#     --weight_epoch 6 \
#     --gs_batch_size 100000 \
#     --gs_iters 100 \
#     --gs_take_best \
#     --eval_ratio 0.1 \
#     --gs_lr 1e-3

# ran gs0 locally, 0.5529916921087135
# Submitted batch job 58758707 # 0.5996211590616561
# Submitted batch job 58758708 # 0.5882946173199274
# Submitted batch job 58758713 # 0.6043499844939226
# Submitted batch job 58758716 # 0.6268429487367216





# noprogram nlp eval gs1 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
    --tag gs1_lr1e-2 \
    --batch_size 4 \
    --flash_attn \
    --weight_dir 0317_noprogram_nlp_llama1b_lora \
    --weight_epoch 6 \
    --gs_batch_size 100000 \
    --gs_iters 1 \
    --gs_take_best \
    --eval_ratio 0.03 \
    --gs_lr 1e-2

# noprogram nlp eval gs5 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
    --tag gs5_lr1e-2 \
    --batch_size 4 \
    --flash_attn \
    --weight_dir 0317_noprogram_nlp_llama1b_lora \
    --weight_epoch 6 \
    --gs_batch_size 100000 \
    --gs_iters 5 \
    --gs_take_best \
    --eval_ratio 0.1 \
    --gs_lr 1e-2

# noprogram nlp eval gs25 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
    --tag gs25_lr1e-2 \
    --batch_size 4 \
    --flash_attn \
    --weight_dir 0317_noprogram_nlp_llama1b_lora \
    --weight_epoch 6 \
    --gs_batch_size 100000 \
    --gs_iters 25 \
    --gs_take_best \
    --eval_ratio 0.1 \
    --gs_lr 1e-2

# noprogram nlp eval gs100 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
    --tag gs100_lr1e-2 \
    --batch_size 4 \
    --flash_attn \
    --weight_dir 0317_noprogram_nlp_llama1b_lora \
    --weight_epoch 6 \
    --gs_batch_size 100000 \
    --gs_iters 100 \
    --gs_take_best \
    --eval_ratio 0.1 \
    --gs_lr 1e-2

# one more round, lr1e-2
# Submitted batch job 58790413
# Submitted batch job 58790414
# Submitted batch job 58790421
# Submitted batch job 58790422
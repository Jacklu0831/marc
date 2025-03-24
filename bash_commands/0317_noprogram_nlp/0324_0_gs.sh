# for gs1, 5
# python make_sbatch.py --ngpu 1 --time 3 --bash_files bash_commands/0317_noprogram_nlp/0324_0_gs.sh

# for gs25
# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_commands/0317_noprogram_nlp/0324_0_gs.sh

# approx: for eval ratio 0.1, gs0 takes literally ~3mins, gs25 takes ~5hrs due to very inefficient implementation
# to start making it more efficient: for each batch, if two demonstration pairs are the same, do not recompute gradient search
# if need more efficiency: caching kv


# # eval gs0
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
#     --tag gs0 \
#     --batch_size 4 \
#     --flash_attn \
#     --weight_dir 0317_noprogram_nlp_llama1b_lora \
#     --weight_epoch 6 \
#     --eval_ratio 0.1

# noprogram nlp eval gs1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
    --tag gs1 \
    --batch_size 4 \
    --flash_attn \
    --weight_dir 0317_noprogram_nlp_llama1b_lora \
    --weight_epoch 6 \
    --gs_batch_size 100000 \
    --gs_iters 1 \
    --gs_take_best \
    --eval_ratio 0.1

# noprogram nlp eval gs5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
    --tag gs5 \
    --batch_size 4 \
    --flash_attn \
    --weight_dir 0317_noprogram_nlp_llama1b_lora \
    --weight_epoch 6 \
    --gs_batch_size 100000 \
    --gs_iters 5 \
    --gs_take_best \
    --eval_ratio 0.1

# noprogram nlp eval gs25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
    --tag gs25 \
    --batch_size 4 \
    --flash_attn \
    --weight_dir 0317_noprogram_nlp_llama1b_lora \
    --weight_epoch 6 \
    --gs_batch_size 100000 \
    --gs_iters 25 \
    --gs_take_best \
    --eval_ratio 0.1





# ran gs0 locally, 0.5529916921087135
# Submitted batch job 58706189
# Submitted batch job 58706190
# Submitted batch job 58706191
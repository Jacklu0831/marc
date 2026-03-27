# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter10 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 10 \
    --ttt_permute_n 1000 \
    --ttt_batch_size 4 \
    --eval_seeds 100

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter50 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 50 \
    --ttt_permute_n 1000 \
    --ttt_batch_size 4 \
    --eval_seeds 100

# prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt_iter10 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 10 \
    --gs_batch_size 4 \
    --eval_seeds 100

# prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt_iter50 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_batch_size 4 \
    --eval_seeds 100

# 18.928497938882735
# 71.51812567029681
# 11.369526931217738

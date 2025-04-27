# test saving ttt ckpts and loading + merging them


# ttt w high LR, save ckpts
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_savettt/test_time_evaluate.py \
    --tag bbh_test_save_ttt \
    --model_name llama8b \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 25 \
    --ttt_lr 1e-2 \
    --ttt_save \
    --eval_ratio 0.01

# 8.641975308641975

# load ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_savettt/test_time_evaluate.py \
    --tag bbh_test_load_ttt \
    --model_name llama8b \
    --ttt_weight_dir eval_bbh_test_save_ttt \
    --eval_ratio 0.01

# 8.641975308641975 # good!
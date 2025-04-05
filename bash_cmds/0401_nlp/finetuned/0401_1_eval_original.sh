# run locally

# nlp gs0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs0 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --eval_seeds 100

# ran locally, score: 0.43772008780802496
# note average over all eval seeds got 0.43453292467372967
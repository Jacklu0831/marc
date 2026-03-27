# bbh ttt iter8 save run2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run2_seed42 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 42

# bbh ttt iter8 save run2 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run2_seed43 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 43

# bbh ttt iter8 save run2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run2_seed44 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 44

# bbh ttt iter8 save run2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run2_seed45 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 45

# bbh ttt iter8 save run2 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run2_seed46 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 46

# 57.39043834887908
# 56.66988331477169
# 56.41203191803037
# 58.37550237486299
# 58.11630792667944
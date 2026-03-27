# bbh ttt iter4 run1 seed46 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter4_run1_seed46_nopermute \
    --ttt_iters 4 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 46

# bbh ttt iter8 run1 seed46 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_run1_seed46_nopermute \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 46

# bbh ttt iter12 run1 seed46 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter12_run1_seed46_nopermute \
    --ttt_iters 12 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 46

# bbh ttt iter16 run1 seed46 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter16_run1_seed46_nopermute \
    --ttt_iters 16 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 46

# bbh ttt iter20 run1 seed46 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter20_run1_seed46_nopermute \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 46

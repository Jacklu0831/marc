# bbh prompt8 lr1e-4 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt8_lr1e-4_tokendrop0.1_seed42 \
    --gs_epochs 8 \
    --gs_lr 1e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --seed 42

# bbh prompt12 lr1e-4 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt12_lr1e-4_tokendrop0.1_seed42 \
    --gs_epochs 12 \
    --gs_lr 1e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --seed 42

# bbh prompt16 lr1e-4 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt16_lr1e-4_tokendrop0.1_seed42 \
    --gs_epochs 16 \
    --gs_lr 1e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --seed 42

# bbh prompt20 lr1e-4 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt20_lr1e-4_tokendrop0.1_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --seed 42

# bbh prompt24 lr1e-4 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt24_lr1e-4_tokendrop0.1_seed42 \
    --gs_epochs 24 \
    --gs_lr 1e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --seed 42

# 51.64628845524462
# 55.511066110292916
# 54.86215942562592
# 55.185270414007526
# 55.926249795450815
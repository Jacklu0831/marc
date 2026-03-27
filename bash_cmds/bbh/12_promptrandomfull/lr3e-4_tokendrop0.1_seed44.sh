# bbh prompt8 lr3e-4 tokendrop0.1 randomfull seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt8_lr3e-4_tokendrop0.1_randomfull_seed44 \
    --gs_epochs 8 \
    --gs_lr 3e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --random_prompt token \
    --seed 44

# bbh prompt12 lr3e-4 tokendrop0.1 randomfull seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt12_lr3e-4_tokendrop0.1_randomfull_seed44 \
    --gs_epochs 12 \
    --gs_lr 3e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --random_prompt token \
    --seed 44

# bbh prompt16 lr3e-4 tokendrop0.1 randomfull seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt16_lr3e-4_tokendrop0.1_randomfull_seed44 \
    --gs_epochs 16 \
    --gs_lr 3e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --random_prompt token \
    --seed 44

# bbh prompt20 lr3e-4 tokendrop0.1 randomfull seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt20_lr3e-4_tokendrop0.1_randomfull_seed44 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --random_prompt token \
    --seed 44

# bbh prompt24 lr3e-4 tokendrop0.1 randomfull seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt24_lr3e-4_tokendrop0.1_randomfull_seed44 \
    --gs_epochs 24 \
    --gs_lr 3e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --random_prompt token \
    --seed 44

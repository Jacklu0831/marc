# mmlu prompt5 lr1e-4 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt5_lr1e-4_tokendrop0.1_random_seed42 \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt10 lr1e-4 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt10_lr1e-4_tokendrop0.1_random_seed42 \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt15 lr1e-4 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt15_lr1e-4_tokendrop0.1_random_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt20 lr1e-4 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt20_lr1e-4_tokendrop0.1_random_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt25 lr1e-4 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt25_lr1e-4_tokendrop0.1_random_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# 34.80897735908164
# 35.01430347283458
# 34.925803342045846
# 35.00176329462589
# 35.09732138822906
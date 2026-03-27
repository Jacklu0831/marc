# mmlu prompt5 lr1e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt5_lr1e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt10 lr1e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt10_lr1e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 10 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt15 lr1e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt15_lr1e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt20 lr1e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt20_lr1e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt25 lr1e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt25_lr1e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# 35.48140117878784
# 34.21543641484877
# 35.8403364014227
# 35.485742272484636
# 35.9555700306247
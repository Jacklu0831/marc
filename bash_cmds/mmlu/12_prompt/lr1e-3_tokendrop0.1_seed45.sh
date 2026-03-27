# mmlu prompt5 lr1e-3 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt5_lr1e-3_tokendrop0.1_seed45 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --ttt_gradient_checkpointing \
    --seed 45

# mmlu prompt10 lr1e-3 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt10_lr1e-3_tokendrop0.1_seed45 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --ttt_gradient_checkpointing \
    --seed 45

# mmlu prompt15 lr1e-3 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt15_lr1e-3_tokendrop0.1_seed45 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --ttt_gradient_checkpointing \
    --seed 45

# mmlu prompt20 lr1e-3 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt20_lr1e-3_tokendrop0.1_seed45 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --ttt_gradient_checkpointing \
    --seed 45

# mmlu prompt25 lr1e-3 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt25_lr1e-3_tokendrop0.1_seed45 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --ttt_gradient_checkpointing \
    --seed 45

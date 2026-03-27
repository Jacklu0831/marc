# mmlu prompt5 lr3e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt5_lr3e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 5 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt10 lr3e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt10_lr3e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 10 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt15 lr3e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt15_lr3e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 15 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt20 lr3e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt20_lr3e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# mmlu prompt25 lr3e-2 tokendrop0.1 random seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt25_lr3e-2_tokendrop0.1_random_seed42 \
    --gs_epochs 25 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42

# 25.365849188796865
# 27.42605924828493
# 28.364358572723795
# 29.227143399695095
# 28.833856945308366
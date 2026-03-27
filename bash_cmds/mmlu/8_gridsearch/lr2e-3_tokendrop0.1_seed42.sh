# mmlu gs5 lr2e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr2e-3_tokendrop0.1_seed42 \
    --gs_epochs 5 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --seed 42

# mmlu gs10 lr2e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr2e-3_tokendrop0.1_seed42 \
    --gs_epochs 10 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --seed 42

# mmlu gs15 lr2e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr2e-3_tokendrop0.1_seed42 \
    --gs_epochs 15 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --seed 42

# mmlu gs20 lr2e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr2e-3_tokendrop0.1_seed42 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --seed 42

# mmlu gs25 lr2e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr2e-3_tokendrop0.1_seed42 \
    --gs_epochs 25 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --seed 42

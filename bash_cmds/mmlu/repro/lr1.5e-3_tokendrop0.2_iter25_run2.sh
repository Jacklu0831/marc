# mmlu gs25 lr1.5e-3 tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.2_seed42_run2 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.2 \
    --gs_float16 \
    --seed 42

# mmlu gs25 lr1.5e-3 tokendrop0.2 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.2_seed43_run2 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.2 \
    --gs_float16 \
    --seed 43

# mmlu gs25 lr1.5e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.2_seed44_run2 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.2 \
    --gs_float16 \
    --seed 44

# mmlu gs25 lr1.5e-3 tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.2_seed45_run2 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.2 \
    --gs_float16 \
    --seed 45

# mmlu gs25 lr1.5e-3 tokendrop0.2 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.2_seed46_run2 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.2 \
    --gs_float16 \
    --seed 46

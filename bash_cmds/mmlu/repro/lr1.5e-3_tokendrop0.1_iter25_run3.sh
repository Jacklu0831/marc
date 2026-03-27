# mmlu gs25 lr1.5e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.1_seed42_run3 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.1 \
    --gs_float16 \
    --seed 42

# mmlu gs25 lr1.5e-3 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.1_seed43_run3 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.1 \
    --gs_float16 \
    --seed 43

# mmlu gs25 lr1.5e-3 tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.1_seed44_run3 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.1 \
    --gs_float16 \
    --seed 44

# mmlu gs25 lr1.5e-3 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.1_seed45_run3 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.1 \
    --gs_float16 \
    --seed 45

# mmlu gs25 lr1.5e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1.5e-3_tokendrop0.1_seed46_run3 \
    --gs_epochs 25 \
    --gs_lr 1.5e-3 \
    --gs_token_dropout 0.1 \
    --gs_float16 \
    --seed 46

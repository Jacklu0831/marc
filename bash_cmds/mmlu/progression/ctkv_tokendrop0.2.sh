# mmlu gs4 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs4_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 4 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --seed 44

# mmlu gs8 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs8_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 8 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --seed 44

# mmlu gs12 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs12_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 12 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --seed 44

# mmlu gs16 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs16_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 16 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --seed 44

# mmlu gs20 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --seed 44

# mmlu gs24 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs24_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 24 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --seed 44

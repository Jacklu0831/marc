# mmlu gs5 lr2e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr2e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 5 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# mmlu gs10 lr2e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr2e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 10 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# mmlu gs15 lr2e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr2e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 15 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# mmlu gs20 lr2e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr2e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# mmlu gs25 lr2e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr2e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 25 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# 43.36090559022704
# 44.75490966784309
# 45.40280402774341
# 44.02158928047276
# 43.07982484791744
# mmlu gs5 lr2e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr2e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 5 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# mmlu gs10 lr2e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr2e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 10 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# mmlu gs15 lr2e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr2e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 15 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# mmlu gs20 lr2e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr2e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# mmlu gs25 lr2e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr2e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 25 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# 42.546658442458856
# 42.445700564122625
# 42.333804976592916
# 41.95989003987937
# 42.18656513081515
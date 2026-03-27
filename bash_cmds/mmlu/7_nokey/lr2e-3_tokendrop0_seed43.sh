# mmlu gs5 lr2e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr2e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 5 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# mmlu gs10 lr2e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr2e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 10 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# mmlu gs15 lr2e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr2e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 15 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# mmlu gs20 lr2e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr2e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# mmlu gs25 lr2e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr2e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 25 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# 42.19621060628805
# 42.182417902470426
# 41.97338486888361
# 42.168582929245474
# 40.94011726189413
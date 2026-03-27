# mmlu gs5 lr2e-3 nokey tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr2e-3_nokey_tokendrop0.1_seed46 \
    --gs_epochs 5 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 46

# mmlu gs10 lr2e-3 nokey tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr2e-3_nokey_tokendrop0.1_seed46 \
    --gs_epochs 10 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 46

# mmlu gs15 lr2e-3 nokey tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr2e-3_nokey_tokendrop0.1_seed46 \
    --gs_epochs 15 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 46

# mmlu gs20 lr2e-3 nokey tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr2e-3_nokey_tokendrop0.1_seed46 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 46

# mmlu gs25 lr2e-3 nokey tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr2e-3_nokey_tokendrop0.1_seed46 \
    --gs_epochs 25 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 46

# 43.00685024959011
# 43.16773792978974
# 42.94123146384697
# 43.09480702346394
# 43.39161484268116
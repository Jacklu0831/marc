# mmlu gs5 lr3e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs10 lr3e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs15 lr3e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs20 lr3e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs25 lr3e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# 42.80420248002637
# 42.79971013935069
# 42.14569783801783
# 41.66520623207436
# 39.9185733242082
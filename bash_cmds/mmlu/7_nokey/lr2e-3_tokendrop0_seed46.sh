# mmlu gs5 lr2e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr2e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 5 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs10 lr2e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr2e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 10 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs15 lr2e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr2e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 15 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs20 lr2e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr2e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs25 lr2e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr2e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 25 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# 42.60409881579634
# 42.711057893231114
# 42.701766051081755
# 41.97120782201763
# 41.853196587006686
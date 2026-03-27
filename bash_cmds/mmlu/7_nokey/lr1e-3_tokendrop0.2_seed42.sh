# mmlu gs5 lr1e-3 nokey tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_nokey_tokendrop0.2_seed42 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 42

# mmlu gs10 lr1e-3 nokey tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_nokey_tokendrop0.2_seed42 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 42

# mmlu gs15 lr1e-3 nokey tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_nokey_tokendrop0.2_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 42

# mmlu gs20 lr1e-3 nokey tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_nokey_tokendrop0.2_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 42

# mmlu gs25 lr1e-3 nokey tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_nokey_tokendrop0.2_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 42

# 41.58806631662334
# 41.593758841344375
# 42.445231203018075
# 42.728335723403525
# 42.82596383460277
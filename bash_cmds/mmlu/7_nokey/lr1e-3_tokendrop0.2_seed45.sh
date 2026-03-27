# mmlu gs5 lr1e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# mmlu gs10 lr1e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# mmlu gs15 lr1e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# mmlu gs20 lr1e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# mmlu gs25 lr1e-3 nokey tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_nokey_tokendrop0.2_seed45 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 45

# 42.529045945850235
# 43.05947115954664
# 43.93859540790923
# 44.04405801428876
# 45.31449597645222
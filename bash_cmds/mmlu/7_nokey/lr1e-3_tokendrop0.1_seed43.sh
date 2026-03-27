# mmlu gs5 lr1e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# mmlu gs10 lr1e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# mmlu gs15 lr1e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# mmlu gs20 lr1e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# mmlu gs25 lr1e-3 nokey tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_nokey_tokendrop0.1_seed43 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 43

# 42.598129124799506
# 42.88246019548255
# 42.536276540241225
# 42.29739480138766
# 42.30211000745697
# mmlu gs5 lr3e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# mmlu gs10 lr3e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# mmlu gs15 lr3e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# mmlu gs20 lr3e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# mmlu gs25 lr3e-3 nokey tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_nokey_tokendrop0_seed43 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 43

# 42.573383117449055
# 42.137811493775764
# 40.84342028583565
# 39.90053405977388
# 39.07919368859872
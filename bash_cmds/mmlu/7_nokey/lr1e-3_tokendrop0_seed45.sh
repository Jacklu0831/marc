# mmlu gs5 lr1e-3 nokey tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_nokey_tokendrop0_seed45 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 45

# mmlu gs10 lr1e-3 nokey tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_nokey_tokendrop0_seed45 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 45

# mmlu gs15 lr1e-3 nokey tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_nokey_tokendrop0_seed45 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 45

# mmlu gs20 lr1e-3 nokey tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_nokey_tokendrop0_seed45 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 45

# mmlu gs25 lr1e-3 nokey tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_nokey_tokendrop0_seed45 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 45

# 42.21893649827903
# 42.89774252134299
# 43.678570491065784
# 43.88374655782909
# 43.45771044113996
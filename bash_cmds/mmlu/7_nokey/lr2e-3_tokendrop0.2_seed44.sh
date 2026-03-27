# mmlu gs5 lr2e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr2e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 5 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# mmlu gs10 lr2e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr2e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 10 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# mmlu gs15 lr2e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr2e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 15 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# mmlu gs20 lr2e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr2e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# mmlu gs25 lr2e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr2e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 25 \
    --gs_lr 2e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# 42.943435237211965
# 43.957694279392584
# 44.21220311832534
# 43.49033735548832
# 43.98547040248846
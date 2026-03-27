# mmlu gs5 lr3e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# mmlu gs10 lr3e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# mmlu gs15 lr3e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# mmlu gs20 lr3e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# mmlu gs25 lr3e-3 nokey tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_nokey_tokendrop0.2_seed44 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 44

# 43.07749590636486
# 43.63663774878269
# 43.8143441607804
# 43.08216197369678
# 42.154414017341914
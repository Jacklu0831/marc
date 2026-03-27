# mmlu gs5 lr3e-3 nokey tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_nokey_tokendrop0.1_seed44 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 44

# mmlu gs10 lr3e-3 nokey tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_nokey_tokendrop0.1_seed44 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 44

# mmlu gs15 lr3e-3 nokey tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_nokey_tokendrop0.1_seed44 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 44

# mmlu gs20 lr3e-3 nokey tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_nokey_tokendrop0.1_seed44 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 44

# mmlu gs25 lr3e-3 nokey tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_nokey_tokendrop0.1_seed44 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.1 \
    --gs_no_key \
    --seed 44

# 43.487405641598805
# 43.43161577605169
# 43.363346878166624
# 42.62117160585216
# 40.74018381833005
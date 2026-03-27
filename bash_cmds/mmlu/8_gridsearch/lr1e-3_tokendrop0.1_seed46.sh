# mmlu gs5 lr1e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_tokendrop0.1_seed46 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --seed 46

# mmlu gs10 lr1e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_tokendrop0.1_seed46 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --seed 46

# mmlu gs15 lr1e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_seed46 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --seed 46

# mmlu gs20 lr1e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_seed46 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --seed 46

# mmlu gs25 lr1e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --seed 46

# 42.84563660644457
# 42.908972566048085
# 42.607956411149885
# 42.712041042248224
# 43.19609546595439
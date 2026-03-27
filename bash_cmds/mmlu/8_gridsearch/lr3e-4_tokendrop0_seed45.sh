# mmlu gs5 lr3e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-4_tokendrop0_seed45 \
    --gs_epochs 5 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu gs10 lr3e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-4_tokendrop0_seed45 \
    --gs_epochs 10 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu gs15 lr3e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-4_tokendrop0_seed45 \
    --gs_epochs 15 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu gs20 lr3e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-4_tokendrop0_seed45 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu gs25 lr3e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-4_tokendrop0_seed45 \
    --gs_epochs 25 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 45

# 41.08373654501908
# 41.551550031939605
# 41.89939479997878
# 42.17889205832518
# 42.60667374968539
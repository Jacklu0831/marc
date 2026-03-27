# mmlu gs5 lr3e-4 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-4_tokendrop0.1_seed45 \
    --gs_epochs 5 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 45

# mmlu gs10 lr3e-4 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-4_tokendrop0.1_seed45 \
    --gs_epochs 10 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 45

# mmlu gs15 lr3e-4 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-4_tokendrop0.1_seed45 \
    --gs_epochs 15 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 45

# mmlu gs20 lr3e-4 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-4_tokendrop0.1_seed45 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 45

# mmlu gs25 lr3e-4 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-4_tokendrop0.1_seed45 \
    --gs_epochs 25 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 45

# 41.378787409226
# 41.53677635125723
# 42.04818007263865
# 42.42095360855639
# 42.88854577664759
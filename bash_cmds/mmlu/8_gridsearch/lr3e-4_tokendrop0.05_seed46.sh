# mmlu gs5 lr3e-4 tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-4_tokendrop0.05_seed46 \
    --gs_epochs 5 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 46

# mmlu gs10 lr3e-4 tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-4_tokendrop0.05_seed46 \
    --gs_epochs 10 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 46

# mmlu gs15 lr3e-4 tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-4_tokendrop0.05_seed46 \
    --gs_epochs 15 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 46

# mmlu gs20 lr3e-4 tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-4_tokendrop0.05_seed46 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 46

# mmlu gs25 lr3e-4 tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-4_tokendrop0.05_seed46 \
    --gs_epochs 25 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 46

# 41.646385766125135
# 42.066192395716655
# 43.05054505685758
# 42.81131651570694
# 42.37161248104906
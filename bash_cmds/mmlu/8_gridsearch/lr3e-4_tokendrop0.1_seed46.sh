# mmlu gs5 lr3e-4 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-4_tokendrop0.1_seed46 \
    --gs_epochs 5 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 46

# mmlu gs10 lr3e-4 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-4_tokendrop0.1_seed46 \
    --gs_epochs 10 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 46

# mmlu gs15 lr3e-4 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-4_tokendrop0.1_seed46 \
    --gs_epochs 15 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 46

# mmlu gs20 lr3e-4 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-4_tokendrop0.1_seed46 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 46

# mmlu gs25 lr3e-4 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-4_tokendrop0.1_seed46 \
    --gs_epochs 25 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.1 \
    --seed 46

# 42.09688177936512
# 42.67931683755443
# 43.097293414866414
# 43.16397983838342
# 42.7618235646121
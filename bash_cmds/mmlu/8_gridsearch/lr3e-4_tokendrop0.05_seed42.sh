# mmlu gs5 lr3e-4 tokendrop0.05 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-4_tokendrop0.05_seed42 \
    --gs_epochs 5 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 42

# mmlu gs10 lr3e-4 tokendrop0.05 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-4_tokendrop0.05_seed42 \
    --gs_epochs 10 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 42

# mmlu gs15 lr3e-4 tokendrop0.05 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-4_tokendrop0.05_seed42 \
    --gs_epochs 15 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 42

# mmlu gs20 lr3e-4 tokendrop0.05 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-4_tokendrop0.05_seed42 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 42

# mmlu gs25 lr3e-4 tokendrop0.05 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-4_tokendrop0.05_seed42 \
    --gs_epochs 25 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 42

# 40.84477433833135
# 41.33492678438043
# 41.48847655439295
# 41.41015414828858
# 41.82312318377032
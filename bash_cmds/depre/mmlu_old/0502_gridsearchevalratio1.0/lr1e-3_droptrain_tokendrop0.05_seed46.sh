# run 2 at a time locally

# mmlu llama8b gs10 lr1e-3 droptrain tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_droptrain_tokendrop0.05_seed46 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_ratio 1.0 \
    --seed 46

# mmlu llama8b gs25 lr1e-3 droptrain tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_droptrain_tokendrop0.05_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_ratio 1.0 \
    --seed 46

# mmlu llama8b gs15 lr1e-3 droptrain tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_droptrain_tokendrop0.05_seed46 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_ratio 1.0 \
    --seed 46

# mmlu llama8b gs20 lr1e-3 droptrain tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_droptrain_tokendrop0.05_seed46 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_ratio 1.0 \
    --seed 46

# need first half
# 42.89111403435876
# 42.79918239648831

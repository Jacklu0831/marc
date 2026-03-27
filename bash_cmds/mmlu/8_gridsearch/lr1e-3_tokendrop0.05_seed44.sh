# mmlu gs5 lr1e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_tokendrop0.05_seed44 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs10 lr1e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_tokendrop0.05_seed44 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs15 lr1e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.05_seed44 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs20 lr1e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.05_seed44 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs25 lr1e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.05_seed44 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# 43.28292332874466
# 43.74214349169951
# 43.628158965387264
# 43.47506380057224
# 43.55066971496749
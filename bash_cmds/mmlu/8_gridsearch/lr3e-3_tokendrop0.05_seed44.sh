# mmlu gs5 lr3e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_tokendrop0.05_seed44 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs10 lr3e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_tokendrop0.05_seed44 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs15 lr3e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_tokendrop0.05_seed44 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs20 lr3e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_tokendrop0.05_seed44 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs25 lr3e-3 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_tokendrop0.05_seed44 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --seed 44

# 44.011130484595455
# 43.07394104052281
# 43.431236494143484
# 41.742160010631885
# 39.664031963222534
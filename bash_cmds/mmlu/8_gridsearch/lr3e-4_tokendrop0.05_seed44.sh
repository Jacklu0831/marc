# mmlu gs5 lr3e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-4_tokendrop0.05_seed44 \
    --gs_epochs 5 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs10 lr3e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-4_tokendrop0.05_seed44 \
    --gs_epochs 10 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs15 lr3e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-4_tokendrop0.05_seed44 \
    --gs_epochs 15 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs20 lr3e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-4_tokendrop0.05_seed44 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu gs25 lr3e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-4_tokendrop0.05_seed44 \
    --gs_epochs 25 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# 42.811498942527955
# 42.908170093767374
# 42.930161010283946
# 43.0572251232099
# 43.40070232699988
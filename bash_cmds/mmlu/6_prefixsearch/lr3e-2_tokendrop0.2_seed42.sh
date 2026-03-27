# mmlu gs5 lr3e-2 tokendrop0.2 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-2_tokendrop0.2_seed42_ntoken32 \
    --gs_epochs 5 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs10 lr3e-2 tokendrop0.2 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-2_tokendrop0.2_seed42_ntoken32 \
    --gs_epochs 10 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs15 lr3e-2 tokendrop0.2 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-2_tokendrop0.2_seed42_ntoken32 \
    --gs_epochs 15 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs20 lr3e-2 tokendrop0.2 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-2_tokendrop0.2_seed42_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs25 lr3e-2 tokendrop0.2 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-2_tokendrop0.2_seed42_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# 40.79048504802595
# 41.068390103297396
# 41.364950044523596
# 41.630689732208005
# 41.32561545581719
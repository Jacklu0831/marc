# mmlu gs5 lr1e-1 tokendrop0.3 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-1_tokendrop0.3_seed42_ntoken32 \
    --gs_epochs 5 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.3 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs10 lr1e-1 tokendrop0.3 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-1_tokendrop0.3_seed42_ntoken32 \
    --gs_epochs 10 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.3 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs15 lr1e-1 tokendrop0.3 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-1_tokendrop0.3_seed42_ntoken32 \
    --gs_epochs 15 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.3 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs20 lr1e-1 tokendrop0.3 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-1_tokendrop0.3_seed42_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.3 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs25 lr1e-1 tokendrop0.3 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-1_tokendrop0.3_seed42_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.3 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

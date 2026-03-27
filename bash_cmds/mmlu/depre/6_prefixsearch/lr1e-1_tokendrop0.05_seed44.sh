# mmlu gs5 lr1e-1 tokendrop0.05 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-1_tokendrop0.05_seed44_ntoken32 \
    --gs_epochs 5 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs10 lr1e-1 tokendrop0.05 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-1_tokendrop0.05_seed44_ntoken32 \
    --gs_epochs 10 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs15 lr1e-1 tokendrop0.05 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-1_tokendrop0.05_seed44_ntoken32 \
    --gs_epochs 15 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs20 lr1e-1 tokendrop0.05 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-1_tokendrop0.05_seed44_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs25 lr1e-1 tokendrop0.05 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-1_tokendrop0.05_seed44_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# 43.03148037970935 <-
# 42.17290687328031
# 40.1773261257111
# 38.53666412665425
# 38.32921155769034
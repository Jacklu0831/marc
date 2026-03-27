# mmlu gs5 lr1e-2 tokendrop0 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-2_tokendrop0_seed44_ntoken32 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs10 lr1e-2 tokendrop0 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-2_tokendrop0_seed44_ntoken32 \
    --gs_epochs 10 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs15 lr1e-2 tokendrop0 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-2_tokendrop0_seed44_ntoken32 \
    --gs_epochs 15 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs20 lr1e-2 tokendrop0 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-2_tokendrop0_seed44_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs25 lr1e-2 tokendrop0 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-2_tokendrop0_seed44_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# 42.80003289977147
# 42.78950951228984
# 42.89229521129674 <-
# 42.70056314959388
# 42.85851328891843
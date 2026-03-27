# mmlu gs5 lr3e-2 tokendrop0.2 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-2_tokendrop0.2_seed44_ntoken32 \
    --gs_epochs 5 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs10 lr3e-2 tokendrop0.2 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-2_tokendrop0.2_seed44_ntoken32 \
    --gs_epochs 10 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs15 lr3e-2 tokendrop0.2 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-2_tokendrop0.2_seed44_ntoken32 \
    --gs_epochs 15 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs20 lr3e-2 tokendrop0.2 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-2_tokendrop0.2_seed44_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# mmlu gs25 lr3e-2 tokendrop0.2 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-2_tokendrop0.2_seed44_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 44

# 42.766508889194014
# 43.53994274258207
# 43.40457818625944
# 43.33221812003483
# 43.06096847579632
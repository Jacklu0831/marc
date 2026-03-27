# mmlu gs5 lr3e-2 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-2_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 5 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 45

# mmlu gs10 lr3e-2 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-2_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 10 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 45

# mmlu gs15 lr3e-2 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-2_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 15 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 45

# mmlu gs20 lr3e-2 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-2_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 45

# mmlu gs25 lr3e-2 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-2_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 45

# 41.28579848455848
# 41.460572581795525
# 41.942758647941915
# 42.09143361301002
# 42.00208690274531
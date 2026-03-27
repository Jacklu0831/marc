# bbh gs8 lr3e-2 tokendrop0.05 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-2_tokendrop0.05_seed45_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs12 lr3e-2 tokendrop0.05 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-2_tokendrop0.05_seed45_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs16 lr3e-2 tokendrop0.05 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-2_tokendrop0.05_seed45_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs20 lr3e-2 tokendrop0.05 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-2_tokendrop0.05_seed45_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 45

# 49.45956643526976
# 50.77761539398367
# 52.93082450371453
# 53.98337595907929
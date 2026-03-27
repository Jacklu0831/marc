# bbh gs8 lr3e-4 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-4_tokendrop0.1_seed45 \
    --gs_epochs 8 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 45

# bbh gs12 lr3e-4 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-4_tokendrop0.1_seed45 \
    --gs_epochs 12 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 45

# bbh gs16 lr3e-4 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-4_tokendrop0.1_seed45 \
    --gs_epochs 16 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 45

# bbh gs20 lr3e-4 tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-4_tokendrop0.1_seed45 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 45

# 52.604889781999766
# 54.89648033126294
# 55.860126659359395
# 56.78495311167945

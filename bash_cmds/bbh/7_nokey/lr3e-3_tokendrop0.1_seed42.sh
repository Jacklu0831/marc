# bbh gs8 lr3e-3 tokendrop0.1 nokey seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-3_tokendrop0.1_nokey_seed42 \
    --gs_epochs 8 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_no_key \
    --seed 42

# bbh gs12 lr3e-3 tokendrop0.1 nokey seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-3_tokendrop0.1_nokey_seed42 \
    --gs_epochs 12 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_no_key \
    --seed 42

# bbh gs16 lr3e-3 tokendrop0.1 nokey seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-3_tokendrop0.1_nokey_seed42 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_no_key \
    --seed 42

# bbh gs20 lr3e-3 tokendrop0.1 nokey seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-3_tokendrop0.1_nokey_seed42 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_no_key \
    --seed 42

# 55.726251840942574
# 55.68994436262478
# 54.086969194894436
# 54.330024750450015
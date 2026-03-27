# bbh gs8 lr2e-3 tokendrop0.05 nokey seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr2e-3_tokendrop0.05_nokey_seed43 \
    --gs_epochs 8 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --gs_no_key \
    --seed 43

# bbh gs12 lr2e-3 tokendrop0.05 nokey seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr2e-3_tokendrop0.05_nokey_seed43 \
    --gs_epochs 12 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --gs_no_key \
    --seed 43

# bbh gs16 lr2e-3 tokendrop0.05 nokey seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr2e-3_tokendrop0.05_nokey_seed43 \
    --gs_epochs 16 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --gs_no_key \
    --seed 43

# bbh gs20 lr2e-3 tokendrop0.05 nokey seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr2e-3_tokendrop0.05_nokey_seed43 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --gs_no_key \
    --seed 43

# 55.299724809602544
# 55.5803782356316
# 54.87633306334582
# 55.41057437031037
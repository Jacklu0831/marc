# bbh gs8 lr2e-3 tokendrop0.2 nokey seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr2e-3_tokendrop0.2_nokey_seed46 \
    --gs_epochs 8 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --gs_no_key \
    --seed 46

# bbh gs12 lr2e-3 tokendrop0.2 nokey seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr2e-3_tokendrop0.2_nokey_seed46 \
    --gs_epochs 12 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --gs_no_key \
    --seed 46

# bbh gs16 lr2e-3 tokendrop0.2 nokey seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr2e-3_tokendrop0.2_nokey_seed46 \
    --gs_epochs 16 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --gs_no_key \
    --seed 46

# bbh gs20 lr2e-3 tokendrop0.2 nokey seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr2e-3_tokendrop0.2_nokey_seed46 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --gs_no_key \
    --seed 46

# 54.73803995282322
# 55.05652151211362
# 55.662409700722385
# 55.347721325372255
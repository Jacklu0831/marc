# bbh gs8 lr3e-3 tokendrop0.1 nokey seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-3_tokendrop0.1_nokey_seed46 \
    --gs_epochs 8 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_no_key \
    --seed 46

# bbh gs12 lr3e-3 tokendrop0.1 nokey seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-3_tokendrop0.1_nokey_seed46 \
    --gs_epochs 12 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_no_key \
    --seed 46

# bbh gs16 lr3e-3 tokendrop0.1 nokey seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-3_tokendrop0.1_nokey_seed46 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_no_key \
    --seed 46

# bbh gs20 lr3e-3 tokendrop0.1 nokey seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-3_tokendrop0.1_nokey_seed46 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_no_key \
    --seed 46

# 54.278481129293816
# 53.71204512506756
# 52.688699383262076
# 53.872111344537814
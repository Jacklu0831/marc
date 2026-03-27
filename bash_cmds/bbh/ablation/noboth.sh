# bbh gs8 lr1e-3 dropnone tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-3_dropnone_tokendrop0_seed44 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --gs_dropout none \
    --seed 44

# bbh gs12 lr1e-3 dropnone tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-3_dropnone_tokendrop0_seed44 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --gs_dropout none \
    --seed 44

# bbh gs16 lr1e-3 dropnone tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-3_dropnone_tokendrop0_seed44 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --gs_dropout none \
    --seed 44

# bbh gs20 lr1e-3 dropnone tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-3_dropnone_tokendrop0_seed44 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --gs_dropout none \
    --seed 44

# 51.02419037790555
# 51.27103911740134
# 51.41051925156028
# 51.07234231411863
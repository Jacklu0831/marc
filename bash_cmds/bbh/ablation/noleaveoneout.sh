# bbh gs8 lr1e-3 dropnone tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-3_dropnone_tokendrop0.1_seed44 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_dropout none \
    --seed 44

# bbh gs12 lr1e-3 dropnone tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-3_dropnone_tokendrop0.1_seed44 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_dropout none \
    --seed 44

# bbh gs16 lr1e-3 dropnone tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-3_dropnone_tokendrop0.1_seed44 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_dropout none \
    --seed 44

# bbh gs20 lr1e-3 dropnone tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-3_dropnone_tokendrop0.1_seed44 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --gs_dropout none \
    --seed 44

# 51.180163951545545
# 53.005276672072334
# 51.96799443461595
# 54.375092142119996
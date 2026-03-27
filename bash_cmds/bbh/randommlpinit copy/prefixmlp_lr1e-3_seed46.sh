# bbh gs8 lr1e-3 randomkv mlp ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_mlp/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-3_randomkv_mlp_ntoken32_seed46 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 46

# bbh gs12 lr1e-3 randomkv mlp ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_mlp/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-3_randomkv_mlp_ntoken32_seed46 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 46

# bbh gs16 lr1e-3 randomkv mlp ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_mlp/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-3_randomkv_mlp_ntoken32_seed46 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 46

# bbh gs20 lr1e-3 randomkv mlp ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_mlp/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-3_randomkv_mlp_ntoken32_seed46 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 46

# bbh gs24 lr1e-3 randomkv mlp ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_mlp/test_time_evaluate.py \
    --tag bbh_gs24_lr1e-3_randomkv_mlp_ntoken32_seed46 \
    --gs_epochs 24 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 46

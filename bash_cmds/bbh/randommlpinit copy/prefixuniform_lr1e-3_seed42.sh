# bbh gs8 lr1e-3 randomkv uniform ntoken32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-3_randomkv_uniform_ntoken32_seed42 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 42

# bbh gs12 lr1e-3 randomkv uniform ntoken32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-3_randomkv_uniform_ntoken32_seed42 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 42

# bbh gs16 lr1e-3 randomkv uniform ntoken32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-3_randomkv_uniform_ntoken32_seed42 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 42

# bbh gs20 lr1e-3 randomkv uniform ntoken32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-3_randomkv_uniform_ntoken32_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 42

# bbh gs24 lr1e-3 randomkv uniform ntoken32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs24_lr1e-3_randomkv_uniform_ntoken32_seed42 \
    --gs_epochs 24 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 42

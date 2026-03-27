# bbh gs8 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 8 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45

# bbh gs12 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 12 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45

# bbh gs16 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45

# bbh gs20 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45

# bbh gs24 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs24_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 24 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45

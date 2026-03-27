# bbh gs1 lr1e-3 randomkv token seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs1_lr1e-3_randomkv_token_seed44 \
    --gs_epochs 1 \
    --gs_lr 1e-9 \
    --gs_dropout none \
    --random_kv token \
    --seed 44

# bbh gs4 lr1e-3 randomkv token seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs4_lr1e-3_randomkv_token_seed44 \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 44

# bbh gs8 lr1e-3 randomkv token seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-3_randomkv_token_seed44 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 44

# bbh gs12 lr1e-3 randomkv token seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-3_randomkv_token_seed44 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 44

# bbh gs16 lr1e-3 randomkv token seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-3_randomkv_token_seed44 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 44

# bbh gs20 lr1e-3 randomkv token seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-3_randomkv_token_seed44 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 44

# bbh gs24 lr1e-3 randomkv token seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs24_lr1e-3_randomkv_token_seed44 \
    --gs_epochs 24 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 44

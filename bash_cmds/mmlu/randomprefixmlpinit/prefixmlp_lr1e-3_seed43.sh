# mmlu llama8b gs5 lr1e-3 randomkv mlp ntoken32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_mlp/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_randomkv_mlp_ntoken32_seed43 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 43

# mmlu llama8b gs10 lr1e-3 randomkv mlp ntoken32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_mlp/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_randomkv_mlp_ntoken32_seed43 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 43

# mmlu llama8b gs15 lr1e-3 randomkv mlp ntoken32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_mlp/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_randomkv_mlp_ntoken32_seed43 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 43

# mmlu llama8b gs20 lr1e-3 randomkv mlp ntoken32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_mlp/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_randomkv_mlp_ntoken32_seed43 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 43

# mmlu llama8b gs25 lr1e-3 randomkv mlp ntoken32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_mlp/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_mlp_ntoken32_seed43 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv mlp \
    --random_kv_ntokens 32 \
    --seed 43
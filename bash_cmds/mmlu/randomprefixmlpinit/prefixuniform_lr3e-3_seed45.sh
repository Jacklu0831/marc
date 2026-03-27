# mmlu llama8b gs5 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45

# mmlu llama8b gs10 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45

# mmlu llama8b gs15 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45

# mmlu llama8b gs20 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45

# mmlu llama8b gs25 lr3e-3 randomkv uniform ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_randomkv_uniform_ntoken32_seed45 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 8 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --seed 45
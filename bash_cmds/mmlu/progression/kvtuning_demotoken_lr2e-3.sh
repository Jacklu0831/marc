# mmlu gs1 lr2e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs2_lr2e-3_randomkv_token_seed45 \
    --gs_epochs 1 \
    --gs_lr 1e-9 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# mmlu gs4 lr2e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs4_lr2e-3_randomkv_token_seed45 \
    --gs_epochs 4 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# mmlu gs8 lr2e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs8_lr2e-3_randomkv_token_seed45 \
    --gs_epochs 8 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# mmlu gs12 lr2e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs12_lr2e-3_randomkv_token_seed45 \
    --gs_epochs 12 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# mmlu gs16 lr2e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs16_lr2e-3_randomkv_token_seed45 \
    --gs_epochs 16 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# mmlu gs20 lr2e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr2e-3_randomkv_token_seed45 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# mmlu gs24 lr2e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs24_lr2e-3_randomkv_token_seed45 \
    --gs_epochs 24 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# # randlora32
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag test \
#     --zero_shot \
#     --gs_epochs 15 \
#     --gs_batch_size 4 \
#     --gs_dropout none \
#     --gs_lora \
#     --gs_lora_randlora \
#     --gs_lora_rank 32 \
#     --gs_lora_alpha 640 \
#     --random_kv uniform \
#     --random_kv_ntokens 0 \
#     --seed 45

# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag test \
#     --zero_shot \
#     --gs_epochs 20 \
#     --gs_batch_size 4 \
#     --gs_dropout none \
#     --gs_lora \
#     --gs_lora_randlora \
#     --gs_lora_rank 32 \
#     --gs_lora_alpha 640 \
#     --random_kv uniform \
#     --random_kv_ntokens 0 \
#     --seed 45

# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag test \
#     --zero_shot \
#     --gs_epochs 25 \
#     --gs_batch_size 4 \
#     --gs_dropout none \
#     --gs_lora \
#     --gs_lora_randlora \
#     --gs_lora_rank 32 \
#     --gs_lora_alpha 640 \
#     --random_kv uniform \
#     --random_kv_ntokens 0 \
#     --seed 45

# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag test \
#     --zero_shot \
#     --gs_epochs 30 \
#     --gs_batch_size 4 \
#     --gs_dropout none \
#     --gs_lora \
#     --gs_lora_randlora \
#     --gs_lora_rank 32 \
#     --gs_lora_alpha 640 \
#     --random_kv uniform \
#     --random_kv_ntokens 0 \
#     --seed 45












# # randlora64
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag test \
#     --zero_shot \
#     --gs_epochs 15 \
#     --gs_batch_size 4 \
#     --gs_dropout none \
#     --gs_lora \
#     --gs_lora_randlora \
#     --gs_lora_rank 64 \
#     --gs_lora_alpha 1280 \
#     --random_kv uniform \
#     --random_kv_ntokens 0 \
#     --seed 45

# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag test \
#     --zero_shot \
#     --gs_epochs 20 \
#     --gs_batch_size 4 \
#     --gs_dropout none \
#     --gs_lora \
#     --gs_lora_randlora \
#     --gs_lora_rank 64 \
#     --gs_lora_alpha 1280 \
#     --random_kv uniform \
#     --random_kv_ntokens 0 \
#     --seed 45

# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag test \
#     --zero_shot \
#     --gs_epochs 25 \
#     --gs_batch_size 4 \
#     --gs_dropout none \
#     --gs_lora \
#     --gs_lora_randlora \
#     --gs_lora_rank 64 \
#     --gs_lora_alpha 1280 \
#     --random_kv uniform \
#     --random_kv_ntokens 0 \
#     --seed 45

# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag test \
#     --zero_shot \
#     --gs_epochs 30 \
#     --gs_batch_size 4 \
#     --gs_dropout none \
#     --gs_lora \
#     --gs_lora_randlora \
#     --gs_lora_rank 64 \
#     --gs_lora_alpha 1280 \
#     --random_kv uniform \
#     --random_kv_ntokens 0 \
#     --seed 45








# randlora128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 128 \
    --gs_lora_alpha 2560 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 128 \
    --gs_lora_alpha 2560 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 128 \
    --gs_lora_alpha 2560 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 128 \
    --gs_lora_alpha 2560 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

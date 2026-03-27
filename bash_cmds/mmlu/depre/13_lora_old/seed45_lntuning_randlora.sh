# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/mmlu/13_lora/seed45_lntuning_randlora.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
# fit 2




# mmlu gs15 lntuning
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_lntuning15_seed45 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lntuning \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs20 lntuning
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_lntuning20_seed45 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lntuning \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs25 lntuning
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_lntuning25_seed45 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lntuning \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs30 lntuning
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_lntuning30_seed45 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lntuning \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45








# mmlu gs15 randlora rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora15_rank16_seed45 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 16 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs20 randlora rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora20_rank16_seed45 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 16 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs25 randlora rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora25_rank16_seed45 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 16 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs30 randlora rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora30_rank16_seed45 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 16 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45








# mmlu gs15 randlora rank32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora15_rank32_seed45 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 32 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs20 randlora rank32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora20_rank32_seed45 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 32 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs25 randlora rank32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora25_rank32_seed45 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 32 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs30 randlora rank32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora30_rank32_seed45 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 32 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45







# mmlu gs15 randlora rank64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora15_rank64_seed45 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 64 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs20 randlora rank64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora20_rank64_seed45 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 64 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs25 randlora rank64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora25_rank64_seed45 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 64 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45

# mmlu gs30 randlora rank64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_randlora30_rank64_seed45 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 64 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 45
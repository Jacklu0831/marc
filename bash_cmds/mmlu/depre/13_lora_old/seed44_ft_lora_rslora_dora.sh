# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/mmlu/13_lora/seed44_ft_lora_rslora_dora.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
# fit 2


# mmlu gs15 ft
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ft15_seed44 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_ft \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs20 ft
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ft20_seed44 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_ft \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs25 ft
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ft25_seed44 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_ft \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs30 ft
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ft30_seed44 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_ft \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44












# mmlu gs15 lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_lora15_seed44 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs20 lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_lora20_seed44 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs25 lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_lora25_seed44 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs30 lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_lora30_seed44 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44












# mmlu gs15 rslora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_rslora15_seed44 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_rslora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs20 rslora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_rslora20_seed44 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_rslora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs25 rslora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_rslora25_seed44 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_rslora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs30 rslora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_rslora30_seed44 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_rslora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44











# mmlu gs15 dora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dora15_seed44 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_dora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs20 dora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dora20_seed44 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_dora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs25 dora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dora25_seed44 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_dora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

# mmlu gs30 dora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dora30_seed44 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_dora \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 44

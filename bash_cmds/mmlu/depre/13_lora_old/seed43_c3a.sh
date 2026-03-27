# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/mmlu/13_lora/seed43_c3a.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# mmlu gs15 c3a lr1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a15_lr1e-1_seed43 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-1 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43

# mmlu gs20 c3a lr1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a20_lr1e-1_seed43 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-1 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43

# mmlu gs25 c3a lr1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a25_lr1e-1_seed43 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-1 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43

# mmlu gs30 c3a lr1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a30_lr1e-1_seed43 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-1 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43










# mmlu gs15 c3a lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a15_lr1e-2_seed43 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-2 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43

# mmlu gs20 c3a lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a20_lr1e-2_seed43 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-2 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43

# mmlu gs25 c3a lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a25_lr1e-2_seed43 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-2 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43

# mmlu gs30 c3a lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a30_lr1e-2_seed43 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-2 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43






# mmlu gs15 c3a lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a15_lr1e-3_seed43 \
    --zero_shot \
    --gs_epochs 15 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-3 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43

# mmlu gs20 c3a lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a20_lr1e-3_seed43 \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-3 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43

# mmlu gs25 c3a lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a25_lr1e-3_seed43 \
    --zero_shot \
    --gs_epochs 25 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-3 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43

# mmlu gs30 c3a lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_c3a30_lr1e-3_seed43 \
    --zero_shot \
    --gs_epochs 30 \
    --gs_batch_size 4 \
    --gs_dropout none \
    --gs_lora \
    --gs_c3a \
    --gs_lora_lr 1e-3 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 43
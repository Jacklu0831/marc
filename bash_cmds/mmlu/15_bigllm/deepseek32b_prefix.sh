# python make_sbatch.py --ngpu 1 --time 12 --gb 64 --single --bash_files bash_cmds/mmlu/15_bigllm/seed42/deepseek32b.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# deepseek 32b prefixntoken15 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_deepseek32b_prefixntoken15_lr3e-3_seed42 \
    --model_name deepseek32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 15 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 10 \
    --seed 42

# deepseek 32b prefixntoken20 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_deepseek32b_prefixntoken20_lr3e-3_seed42 \
    --model_name deepseek32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 20 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 10 \
    --seed 42

# deepseek 32b prefixntoken25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_deepseek32b_prefixntoken25_lr3e-3_seed42 \
    --model_name deepseek32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 25 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 10 \
    --seed 42

# deepseek 32b prefixntoken30 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_deepseek32b_prefixntoken30_lr3e-3_seed42 \
    --model_name deepseek32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 30 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 10 \
    --seed 42

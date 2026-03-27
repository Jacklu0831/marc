# python make_sbatch.py --ngpu 1 --time 12 --gb 64 --single --bash_files bash_cmds/bbh/15_bigllm/deepseek14b.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# deepseek 14b prefixntoken15 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_deepseek14b_prefixntoken15_lr3e-3_seed42 \
    --model_name deepseek14b \
    --batch_size 4 \
    --gs_epochs 15 \
    --gs_batch_size 2 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42

# deepseek 14b prefixntoken20 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_deepseek14b_prefixntoken20_lr3e-3_seed42 \
    --model_name deepseek14b \
    --batch_size 4 \
    --gs_epochs 20 \
    --gs_batch_size 2 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42

# deepseek 14b prefixntoken25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_deepseek14b_prefixntoken25_lr3e-3_seed42 \
    --model_name deepseek14b \
    --batch_size 4 \
    --gs_epochs 25 \
    --gs_batch_size 2 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42

# deepseek 14b prefixntoken30 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_deepseek14b_prefixntoken30_lr3e-3_seed42 \
    --model_name deepseek14b \
    --batch_size 4 \
    --gs_epochs 30 \
    --gs_batch_size 2 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42

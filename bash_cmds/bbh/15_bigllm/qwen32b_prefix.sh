# python make_sbatch.py --ngpu 1 --time 12 --gb 128 --single --bash_files bash_cmds/bbh/15_bigllm/qwen32b.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# qwen 32b prefixntoken15 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_qwen32b_prefixntoken15_lr3e-3_seed42 \
    --model_name qwen32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 15 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42

# qwen 32b prefixntoken20 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_qwen32b_prefixntoken20_lr3e-3_seed42 \
    --model_name qwen32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 20 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42

# qwen 32b prefixntoken25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_qwen32b_prefixntoken25_lr3e-3_seed42 \
    --model_name qwen32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 25 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42

# qwen 32b prefixntoken30 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_qwen32b_prefixntoken30_lr3e-3_seed42 \
    --model_name qwen32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 30 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42

# python make_sbatch.py --ngpu 1 --time 3 --single --bash_files bash_cmds/0401_bbh/0425_randomsearchntoken/0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# bbh llama8b gs25 lr1e-2 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-2_randomkv_token_ntoken32 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 45

# bbh llama8b gs50 lr1e-2 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-2_randomkv_token_ntoken32 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 45

# bbh llama8b gs75 lr1e-2 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs75_lr1e-2_randomkv_token_ntoken32 \
    --model_name llama8b \
    --gs_epochs 75 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 45

# bbh llama8b gs100 lr1e-2 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-2_randomkv_token_ntoken32 \
    --model_name llama8b \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 45

# Submitted batch job 59824122
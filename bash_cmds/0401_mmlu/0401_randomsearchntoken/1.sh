# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_mmlu/0401_randomsearchntoken/1.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs25 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --random_kv token \
    --random_kv_ntokens 32

# mmlu llama8b gs50 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs50_lr1e-3_randomkv_token_ntoken32 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --random_kv token \
    --random_kv_ntokens 32

# mmlu llama8b gs75 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs75_lr1e-3_randomkv_token_ntoken32 \
    --gs_epochs 75 \
    --gs_lr 1e-3 \
    --random_kv token \
    --random_kv_ntokens 32

# mmlu llama8b gs100 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs100_lr1e-3_randomkv_token_ntoken32 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --random_kv token \
    --random_kv_ntokens 32

# 41.10646975008968
# 36.5159473655885
# 35.19773207761037
# out of time, no matter
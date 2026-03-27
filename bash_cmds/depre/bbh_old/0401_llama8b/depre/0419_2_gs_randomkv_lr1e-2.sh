# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_bbh/llama8b/0419_2_gs_randomkv_lr1e-2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)








# bbh llama8b gs25 lr1e-2 randomkv normal
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-2_randomkv_normal \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv normal

# bbh llama8b gs50 lr1e-2 randomkv normal
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-2_randomkv_normal \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv normal

# bbh llama8b gs75 lr1e-2 randomkv normal
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs75_lr1e-2_randomkv_normal \
    --model_name llama8b \
    --gs_epochs 75 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv normal

# bbh llama8b gs100 lr1e-2 randomkv normal
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-2_randomkv_normal \
    --model_name llama8b \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv normal









# bbh llama8b gs25 lr1e-2 randomkv ntoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-2_randomkv_ntoken \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv token

# bbh llama8b gs50 lr1e-2 randomkv ntoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-2_randomkv_ntoken \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv token

# bbh llama8b gs75 lr1e-2 randomkv ntoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs75_lr1e-2_randomkv_ntoken \
    --model_name llama8b \
    --gs_epochs 75 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv token

# bbh llama8b gs100 lr1e-2 randomkv ntoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-2_randomkv_ntoken \
    --model_name llama8b \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --random_kv token

# Submitted batch job 59479447

# normal
# 16.58784449121771
# 13.376198564799427
# 11.107187696170747
# 11.485404896421846

# token
# 32.678907721280595
# 31.66742277576442
# 31.100899242308348
# connection
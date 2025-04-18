# python make_sbatch.py --ngpu 1 --time 2 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_6_compression.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# bbh llama8b compression0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_compression0.9 \
    --model_name llama8b \
    --compression_ratio 0.9

# bbh llama8b compression0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_compression0.75 \
    --model_name llama8b \
    --compression_ratio 0.75

# bbh llama8b compression0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_compression0.5 \
    --model_name llama8b \
    --compression_ratio 0.5

# bbh llama8b compression0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_compression0.25 \
    --model_name llama8b \
    --compression_ratio 0.25

# Submitted batch job 59408827
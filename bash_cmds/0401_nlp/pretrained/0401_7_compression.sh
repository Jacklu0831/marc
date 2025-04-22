# python make_sbatch.py --ngpu 1 --time 2 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0401_7_compression.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# nlp compression0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_compression0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --compression_ratio 0.9 \
    --eval_seeds 100

# nlp compression0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_compression0.75 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --compression_ratio 0.75 \
    --eval_seeds 100

# nlp compression0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_compression0.5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --compression_ratio 0.5 \
    --eval_seeds 100

# nlp compression0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_compression0.25 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --compression_ratio 0.25 \
    --eval_seeds 100

# Submitted batch job 59510093

# 0.36759887629687793
# 0.36377696097189727
# 0.375769815872376 <-
# 0.3522207050434485

# so far 0.375769815872376 with compression 0.5, interesting
# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --bash_files bash_cmds/nlp/13_lora/randlora_rank16.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs150 randlora rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_randlora150_rank16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --zero_shot \
    --gs_epochs 150 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 16 \
    --gs_lora_alpha 320 \
    --random_kv uniform \
    --random_kv_ntokens 0

# nlp gs200 randlora rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_randlora200_rank16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --zero_shot \
    --gs_epochs 200 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 16 \
    --gs_lora_alpha 320 \
    --random_kv uniform \
    --random_kv_ntokens 0

# nlp gs250 randlora rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_randlora250_rank16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --zero_shot \
    --gs_epochs 250 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 16 \
    --gs_lora_alpha 320 \
    --random_kv uniform \
    --random_kv_ntokens 0

# nlp gs300 randlora rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_randlora300_rank16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --zero_shot \
    --gs_epochs 300 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 16 \
    --gs_lora_alpha 320 \
    --random_kv uniform \
    --random_kv_ntokens 0

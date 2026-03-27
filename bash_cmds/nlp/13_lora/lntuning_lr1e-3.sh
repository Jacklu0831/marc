# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --bash_files bash_cmds/nlp/13_lora/lntuning_lr1e-3.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs150 lntuning
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_lntuning150 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --zero_shot \
    --gs_epochs 150 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --gs_lntuning \
    --random_kv uniform \
    --random_kv_ntokens 0

# nlp gs200 lntuning
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_lntuning200 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --zero_shot \
    --gs_epochs 200 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --gs_lntuning \
    --random_kv uniform \
    --random_kv_ntokens 0

# nlp gs250 lntuning
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_lntuning250 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --zero_shot \
    --gs_epochs 250 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --gs_lntuning \
    --random_kv uniform \
    --random_kv_ntokens 0

# nlp gs300 lntuning
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_lntuning300 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --zero_shot \
    --gs_epochs 300 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --gs_lntuning \
    --random_kv uniform \
    --random_kv_ntokens 0

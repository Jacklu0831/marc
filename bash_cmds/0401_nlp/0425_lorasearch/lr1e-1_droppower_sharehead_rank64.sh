# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_lorasearch/lr1e-1_droppower_sharehead_rank64.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)





# nlp gs25 lr1e-1 droppower sharehead rank64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-1_droppower_sharehead_rank64 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-1 \
    --gs_dropout power \
    --gs_prefix_lora \
    --gs_prefix_lora_rank 64 \
    --gs_prefix_lora_share_head \
    --eval_seeds 100

# nlp gs50 lr1e-1 droppower sharehead rank64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-1_droppower_sharehead_rank64 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-1 \
    --gs_dropout power \
    --gs_prefix_lora \
    --gs_prefix_lora_rank 64 \
    --gs_prefix_lora_share_head \
    --eval_seeds 100

# nlp gs75 lr1e-1 droppower sharehead rank64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs75_lr1e-1_droppower_sharehead_rank64 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 75 \
    --gs_lr 1e-1 \
    --gs_dropout power \
    --gs_prefix_lora \
    --gs_prefix_lora_rank 64 \
    --gs_prefix_lora_share_head \
    --eval_seeds 100

# nlp gs100 lr1e-1 droppower sharehead rank64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-1_droppower_sharehead_rank64 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-1 \
    --gs_dropout power \
    --gs_prefix_lora \
    --gs_prefix_lora_rank 64 \
    --gs_prefix_lora_share_head \
    --eval_seeds 100

# Submitted batch job 59815066

# 0.3282653198285108
# 0.33135711452900674
# 0.33119721805684765
# 0.33197163400020835
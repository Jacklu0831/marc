# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_lorasearch/lr1e-2_droptrain_nosharehead_rank16.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)





# nlp gs25 lr1e-2 droptrain nosharehead rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-2_droptrain_nosharehead_rank16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_prefix_lora \
    --gs_prefix_lora_rank 16 \
    --eval_seeds 100

# nlp gs50 lr1e-2 droptrain nosharehead rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-2_droptrain_nosharehead_rank16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_prefix_lora \
    --gs_prefix_lora_rank 16 \
    --eval_seeds 100

# nlp gs75 lr1e-2 droptrain nosharehead rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs75_lr1e-2_droptrain_nosharehead_rank16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 75 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_prefix_lora \
    --gs_prefix_lora_rank 16 \
    --eval_seeds 100

# nlp gs100 lr1e-2 droptrain nosharehead rank16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-2_droptrain_nosharehead_rank16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_prefix_lora \
    --gs_prefix_lora_rank 16 \
    --eval_seeds 100

# Submitted batch job 59815071
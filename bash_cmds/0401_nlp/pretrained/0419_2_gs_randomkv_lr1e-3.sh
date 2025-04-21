# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0419_2_gs_randomkv_lr1e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr1e-3 randomkv normal
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_randomkv_normal \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --random_kv normal \
    --eval_seeds 100

# nlp gs25 lr1e-3 randomkv normal
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_randomkv_normal \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --random_kv normal \
    --eval_seeds 100

# nlp gs100 lr1e-3 randomkv normal
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_randomkv_normal \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --random_kv normal \
    --eval_seeds 100

# nlp gs250 lr1e-3 randomkv normal
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_randomkv_normal \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --random_kv normal \
    --eval_seeds 100








# nlp gs5 lr1e-3 randomkv ntoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_randomkv_ntoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --random_kv ntoken \
    --eval_seeds 100

# nlp gs25 lr1e-3 randomkv ntoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_randomkv_ntoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --random_kv ntoken \
    --eval_seeds 100

# nlp gs100 lr1e-3 randomkv ntoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_randomkv_ntoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --random_kv ntoken \
    --eval_seeds 100

# nlp gs250 lr1e-3 randomkv ntoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_randomkv_ntoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --random_kv ntoken \
    --eval_seeds 100

# Submitted batch job 59510243
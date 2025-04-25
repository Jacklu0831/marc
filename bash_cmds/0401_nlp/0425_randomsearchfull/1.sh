# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_randomsearchfull/1.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs100 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --random_kv token

# nlp gs200 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --random_kv token

# nlp gs300 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs300_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 300 \
    --gs_lr 1e-3 \
    --random_kv token

# nlp gs400 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs400_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 400 \
    --gs_lr 1e-3 \
    --random_kv token

# Submitted batch job 59760874
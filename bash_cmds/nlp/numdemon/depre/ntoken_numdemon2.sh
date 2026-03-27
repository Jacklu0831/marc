# python make_sbatch.py --ngpu 1 --time 10 --rtx8000 --single --bash_files bash_cmds/nlp/numdemon/ntoken_numdemon2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs10 lr1e-3 randomkv token ntoken32 ndemon2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs10_lr1e-3_randomkv_token_ntoken32_ndemon2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 2

# nlp gs25 lr1e-3 randomkv token ntoken32 ndemon2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_randomkv_token_ntoken32_ndemon2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 2

# nlp gs50 lr1e-3 randomkv token ntoken32 ndemon2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_randomkv_token_ntoken32_ndemon2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 2

# nlp gs100 lr1e-3 randomkv token ntoken32 ndemon2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_randomkv_token_ntoken32_ndemon2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 2

# Submitted batch job 60534968

# 0.35402773481793204
# 0.3549928888094055
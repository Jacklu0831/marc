# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/nlp/3_randomsearchntoken/lr3e-3.sh

# nlp gs100 lr3e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# nlp gs150 lr3e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr3e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# nlp gs200 lr3e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr3e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# nlp gs250 lr3e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# Submitted batch job 60461654
# Submitted batch job 60461655
# Submitted batch job 60461656
# Submitted batch job 60461657

# 0.420999428184113
# 0.4205874441691039
# 0.4191635419703282
# 0.4197591156568862
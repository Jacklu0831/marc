# python make_sbatch.py --ngpu 1 --time 15 --rtx8000 --bash_files bash_cmds/nlp/3_randomsearchntoken/lr1e-3.sh

# nlp gs1 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs1_lr1e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_lr 1e-9 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# nlp gs50 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# nlp gs100 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# nlp gs150 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# # nlp gs200 lr1e-3 randomkv token ntoken32
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag nlp_gs200_lr1e-3_randomkv_token_ntoken32 \
#     --weight_dir nlp_pretrained \
#     --weight_epoch 0 \
#     --gs_epochs 200 \
#     --gs_lr 1e-3 \
#     --gs_dropout none \
#     --random_kv token \
#     --random_kv_ntokens 32

# nlp gs250 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# # nlp gs300 lr1e-3 randomkv token ntoken32
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag nlp_gs300_lr1e-3_randomkv_token_ntoken32 \
#     --weight_dir nlp_pretrained \
#     --weight_epoch 0 \
#     --gs_epochs 300 \
#     --gs_lr 1e-3 \
#     --gs_dropout none \
#     --random_kv token \
#     --random_kv_ntokens 32

# nlp gs350 lr1e-3 randomkv token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs350_lr1e-3_randomkv_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 350 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32

# # nlp gs400 lr1e-3 randomkv token ntoken32
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag nlp_gs400_lr1e-3_randomkv_token_ntoken32 \
#     --weight_dir nlp_pretrained \
#     --weight_epoch 0 \
#     --gs_epochs 400 \
#     --gs_lr 1e-3 \
#     --gs_dropout none \
#     --random_kv token \
#     --random_kv_ntokens 32


# Submitted batch job 60517624
# Submitted batch job 60517625
# Submitted batch job 60517626
# Submitted batch job 60517627
# Submitted batch job 60096910
# Submitted batch job 60517628
# Submitted batch job 60096911
# Submitted batch job 60517629
# Submitted batch job 60096912

# 0:
# 50:
# 100:
# 150:
# 200: 0.4198666045619307
# 250:
# 300: 0.42219849283782745
# 350:
# 400: 0.4220260519101588
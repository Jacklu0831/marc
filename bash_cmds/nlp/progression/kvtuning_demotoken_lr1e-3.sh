# python make_sbatch.py --ngpu 1 --time 15 --rtx8000 --bash_files bash_cmds/nlp/4_randomsearchfull/lr1e-3.sh

# nlp gs1 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs1_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_lr 1e-9 \
    --gs_dropout none \
    --random_kv token

# nlp gs50 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token

# nlp gs100 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token

# nlp gs150 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token

# # nlp gs200 lr1e-3 randomkv token
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag nlp_gs200_lr1e-3_randomkv_token \
#     --weight_dir nlp_pretrained \
#     --weight_epoch 0 \
#     --gs_epochs 200 \
#     --gs_lr 1e-3 \
#     --gs_dropout none \
#     --random_kv token

# nlp gs250 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token

# # nlp gs300 lr1e-3 randomkv token
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag nlp_gs300_lr1e-3_randomkv_token \
#     --weight_dir nlp_pretrained \
#     --weight_epoch 0 \
#     --gs_epochs 300 \
#     --gs_lr 1e-3 \
#     --gs_dropout none \
#     --random_kv token

# nlp gs350 lr1e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs350_lr1e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 350 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token

# # nlp gs400 lr1e-3 randomkv token
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag nlp_gs400_lr1e-3_randomkv_token \
#     --weight_dir nlp_pretrained \
#     --weight_epoch 0 \
#     --gs_epochs 400 \
#     --gs_lr 1e-3 \
#     --gs_dropout none \
#     --random_kv token





# Submitted batch job 60517651
# Submitted batch job 60517652
# Submitted batch job 60517653
# Submitted batch job 60517654
# Submitted batch job 60096913
# Submitted batch job 60517655
# Submitted batch job 60096914
# Submitted batch job 60517656
# Submitted batch job 60096915

# 200: 0.4094654234163779
# 300: 0.4102568161599215
# 400: 0.4106588700592363

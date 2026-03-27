# python make_sbatch.py --ngpu 1 --time 16 --rtx8000 --bash_files bash_cmds/nlp/4_randomsearchfull/lr3e-3.sh

# nlp gs100 lr3e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token

# nlp gs150 lr3e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr3e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token

# nlp gs200 lr3e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr3e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token

# nlp gs250 lr3e-3 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_randomkv_token \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token

# Submitted batch job 60461660
# Submitted batch job 60461661
# Submitted batch job 60461662
# Submitted batch job 60461663

# 0.4102928675470662
# 0.41006970402251924
# 0.4073968890282421
# 0.4083840273499774
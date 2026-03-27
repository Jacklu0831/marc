# python make_sbatch.py --ngpu 1 --time 20 --rtx8000 --single --bash_files bash_cmds/nlp/progression/ctkv_tokendrop0.1.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs1 lr1e-3 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs1_lr1e-3_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_lr 1e-9 \
    --gs_dropout train \
    --gs_token_dropout 0.1

# nlp gs50 lr1e-3 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1

# nlp gs100 lr1e-3 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1

# nlp gs150 lr1e-3 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1

# nlp gs200 lr1e-3 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1

# nlp gs250 lr1e-3 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1

# 0: icl performance
# 50: 0.41667619558844615
# 100: 0.43974988056040204
# 150: 0.4394439783866293
# 200: 0.4381509583470494
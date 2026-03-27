# python make_sbatch.py --ngpu 1 --time 20 --rtx8000 --single --bash_files bash_cmds/nlp/8_gridsearch_150_to_250_CHECK/lr1e-4_droptrain_tokendrop0.1.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs150 lr1e-4 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-4_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1

# nlp gs200 lr1e-4 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-4_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1

# nlp gs250 lr1e-4 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-4_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1

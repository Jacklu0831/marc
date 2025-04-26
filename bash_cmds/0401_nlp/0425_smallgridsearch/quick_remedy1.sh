# python make_sbatch.py --ngpu 1 --time 16 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_smallgridsearch/quick_remedy1.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs100 lr1e-3 droptrain tokendrop0.01 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_tokendrop0.01_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 0.0

# nlp gs100 lr1e-3 droptrain tokendrop0.01 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_tokendrop0.01_lambda1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 1e-2

# nlp gs100 lr1e-3 droptrain tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_tokendrop0.05_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --gs_lambda_param_sqr 0.0

# nlp gs100 lr1e-3 droptrain tokendrop0.05 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_tokendrop0.05_lambda1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --gs_lambda_param_sqr 1e-2

# Submitted batch job 59808835
# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_smallgridsearchmoreiter/lr5e-3_droptrain_tokendrop0.01_lambda0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs125 lr5e-3 droptrain tokendrop0.01 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs125_lr5e-3_droptrain_tokendrop0.01_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 125 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 0.0

# nlp gs150 lr5e-3 droptrain tokendrop0.01 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr5e-3_droptrain_tokendrop0.01_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 0.0

# Submitted batch job 59798738
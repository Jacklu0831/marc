# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_smallgridsearch/lr5e-3_droppower_tokendrop0_lambda0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs25 lr5e-3 droppower tokendrop0 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr5e-3_droppower_tokendrop0_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0 \
    --gs_lambda_param_sqr 0.0

# nlp gs50 lr5e-3 droppower tokendrop0 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr5e-3_droppower_tokendrop0_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0 \
    --gs_lambda_param_sqr 0.0

# nlp gs75 lr5e-3 droppower tokendrop0 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs75_lr5e-3_droppower_tokendrop0_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 75 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0 \
    --gs_lambda_param_sqr 0.0

# nlp gs100 lr5e-3 droppower tokendrop0 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr5e-3_droppower_tokendrop0_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0 \
    --gs_lambda_param_sqr 0.0

# Submitted batch job 59822307

# 0.385662986714403
# 0.39711807042765107
# 0.4095238147156729
# 0.4222531980139593
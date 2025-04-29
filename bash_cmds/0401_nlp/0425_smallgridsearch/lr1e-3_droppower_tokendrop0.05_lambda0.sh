# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_smallgridsearch/lr1e-3_droppower_tokendrop0.05_lambda0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs25 lr1e-3 droppower tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_droppower_tokendrop0.05_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.05 \
    --gs_lambda_param_sqr 0.0

# nlp gs50 lr1e-3 droppower tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_droppower_tokendrop0.05_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.05 \
    --gs_lambda_param_sqr 0.0

# nlp gs75 lr1e-3 droppower tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs75_lr1e-3_droppower_tokendrop0.05_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 75 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.05 \
    --gs_lambda_param_sqr 0.0

# nlp gs100 lr1e-3 droppower tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droppower_tokendrop0.05_lambda0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.05 \
    --gs_lambda_param_sqr 0.0

# Submitted batch job 59761503

# 0.37653104243266017
# 0.3877997272598877
# 0.3901124396802108
# 0.3883125773035562
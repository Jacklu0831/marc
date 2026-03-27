# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_smallgridsearch/lr5e-3_droppower_tokendrop0.1_lambda1e-1.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs25 lr5e-3 droppower tokendrop0.1 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr5e-3_droppower_tokendrop0.1_lambda1e-1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 1e-1

# nlp gs50 lr5e-3 droppower tokendrop0.1 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr5e-3_droppower_tokendrop0.1_lambda1e-1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 1e-1

# nlp gs75 lr5e-3 droppower tokendrop0.1 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs75_lr5e-3_droppower_tokendrop0.1_lambda1e-1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 75 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 1e-1

# nlp gs100 lr5e-3 droppower tokendrop0.1 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr5e-3_droppower_tokendrop0.1_lambda1e-1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 1e-1

# Submitted batch job 59822512

# 0.37368218110300117
# 0.3753147850868981
# 0.37539435869830096
# 0.37322603149915595
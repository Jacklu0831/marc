# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_smallgridsearchmoreiter/lr1e-3_droppower_tokendrop0.01_lambda1e-2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs125 lr1e-3 droppower tokendrop0.01 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs125_lr1e-3_droppower_tokendrop0.01_lambda1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 125 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 1e-2

# nlp gs150 lr1e-3 droppower tokendrop0.01 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_droppower_tokendrop0.01_lambda1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 1e-2

# Submitted batch job 59808748
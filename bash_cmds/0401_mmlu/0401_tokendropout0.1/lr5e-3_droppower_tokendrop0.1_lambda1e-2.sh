# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_mmlu/0401_tokendropout0.1/lr5e-3_droppower_tokendrop0.1_lambda1e-2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs5 lr5e-3 droppower tokendrop0.1 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr5e-3_droppower_tokendrop0.1_lambda1e-2 \
    --gs_epochs 5 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 1e-2

# mmlu llama8b gs10 lr5e-3 droppower tokendrop0.1 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr5e-3_droppower_tokendrop0.1_lambda1e-2 \
    --gs_epochs 10 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 1e-2

# mmlu llama8b gs15 lr5e-3 droppower tokendrop0.1 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr5e-3_droppower_tokendrop0.1_lambda1e-2 \
    --gs_epochs 15 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 1e-2

# mmlu llama8b gs20 lr5e-3 droppower tokendrop0.1 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr5e-3_droppower_tokendrop0.1_lambda1e-2 \
    --gs_epochs 20 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 1e-2

# mmlu llama8b gs25 lr5e-3 droppower tokendrop0.1 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr5e-3_droppower_tokendrop0.1_lambda1e-2 \
    --gs_epochs 25 \
    --gs_lr 5e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 1e-2

# 41.60619198175083
# 40.52662783643295
# 38.34522070250947
# 39.30602408201061
# 37.66870242230853
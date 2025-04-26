# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_mmlu/0401_smallgridsearch/lr1e-3_droppower_tokendrop0.01_lambda0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs5 lr1e-3 droppower tokendrop0.01 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_droppower_tokendrop0.01_lambda0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 0.0

# mmlu llama8b gs10 lr1e-3 droppower tokendrop0.01 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_droppower_tokendrop0.01_lambda0 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 0.0

# mmlu llama8b gs15 lr1e-3 droppower tokendrop0.01 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_droppower_tokendrop0.01_lambda0 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 0.0

# mmlu llama8b gs20 lr1e-3 droppower tokendrop0.01 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_droppower_tokendrop0.01_lambda0 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 0.0

# mmlu llama8b gs25 lr1e-3 droppower tokendrop0.01 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_droppower_tokendrop0.01_lambda0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 0.0

# 42.221710579579124
# 42.35329801549306
# 42.03552514439699
# 42.559231137369586
# 42.91364188886267
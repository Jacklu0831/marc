# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_mmlu/0401_tokendropout0.1/lr5e-3_droptrain_tokendrop0.1_lambda0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs5 lr5e-3 droptrain tokendrop0.1 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr5e-3_droptrain_tokendrop0.1_lambda0 \
    --gs_epochs 5 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 0.0

# mmlu llama8b gs10 lr5e-3 droptrain tokendrop0.1 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr5e-3_droptrain_tokendrop0.1_lambda0 \
    --gs_epochs 10 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 0.0

# mmlu llama8b gs15 lr5e-3 droptrain tokendrop0.1 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr5e-3_droptrain_tokendrop0.1_lambda0 \
    --gs_epochs 15 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 0.0

# mmlu llama8b gs20 lr5e-3 droptrain tokendrop0.1 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr5e-3_droptrain_tokendrop0.1_lambda0 \
    --gs_epochs 20 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 0.0

# mmlu llama8b gs25 lr5e-3 droptrain tokendrop0.1 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr5e-3_droptrain_tokendrop0.1_lambda0 \
    --gs_epochs 25 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 0.0

# 40.19635367362696
# 39.40654215221892
# 33.94751501167725
# 31.577063514983045
# 30.88864916421846
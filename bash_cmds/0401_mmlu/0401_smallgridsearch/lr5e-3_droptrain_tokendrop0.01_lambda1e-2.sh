# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_mmlu/0401_smallgridsearch/lr5e-3_droptrain_tokendrop0.01_lambda1e-2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs5 lr5e-3 droptrain tokendrop0.01 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr5e-3_droptrain_tokendrop0.01_lambda1e-2 \
    --gs_epochs 5 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 1e-2

# mmlu llama8b gs10 lr5e-3 droptrain tokendrop0.01 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr5e-3_droptrain_tokendrop0.01_lambda1e-2 \
    --gs_epochs 10 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 1e-2

# mmlu llama8b gs15 lr5e-3 droptrain tokendrop0.01 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr5e-3_droptrain_tokendrop0.01_lambda1e-2 \
    --gs_epochs 15 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 1e-2

# mmlu llama8b gs20 lr5e-3 droptrain tokendrop0.01 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr5e-3_droptrain_tokendrop0.01_lambda1e-2 \
    --gs_epochs 20 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 1e-2

# mmlu llama8b gs25 lr5e-3 droptrain tokendrop0.01 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr5e-3_droptrain_tokendrop0.01_lambda1e-2 \
    --gs_epochs 25 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_lambda_param_sqr 1e-2

# 41.7488928705857
# 40.85679699498458
# 40.04543151412074
# 39.053216473181536
# 41.09226607226743
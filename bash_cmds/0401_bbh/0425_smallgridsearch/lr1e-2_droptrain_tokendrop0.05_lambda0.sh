# python make_sbatch.py --ngpu 1 --time 2 --single --bash_files bash_cmds/0401_bbh/0425_smallgridsearch/lr1e-2_droptrain_tokendrop0.05_lambda0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)






# bbh llama8b gs5 lr1e-2 droptrain tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-2_droptrain_tokendrop0.05_lambda0 \
    --model_name llama8b \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --gs_lambda_param_sqr 0.0 \
    --seed 45

# bbh llama8b gs10 lr1e-2 droptrain tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-2_droptrain_tokendrop0.05_lambda0 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --gs_lambda_param_sqr 0.0 \
    --seed 45

# bbh llama8b gs15 lr1e-2 droptrain tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs15_lr1e-2_droptrain_tokendrop0.05_lambda0 \
    --model_name llama8b \
    --gs_epochs 15 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --gs_lambda_param_sqr 0.0 \
    --seed 45

# bbh llama8b gs20 lr1e-2 droptrain tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-2_droptrain_tokendrop0.05_lambda0 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --gs_lambda_param_sqr 0.0 \
    --seed 45

# bbh llama8b gs25 lr1e-2 droptrain tokendrop0.05 lambda0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-2_droptrain_tokendrop0.05_lambda0 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --gs_lambda_param_sqr 0.0 \
    --seed 45

# 46.132928375367726
# 38.89373765827737
# 35.09694021990467
# 32.94513725138517
# 35.301447948922196
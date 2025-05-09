# python make_sbatch.py --ngpu 1 --time 2 --single --bash_files bash_cmds/0401_bbh/0425_smallgridsearch/lr5e-3_droptrain_tokendrop0.01_lambda1e-1.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)






# bbh llama8b gs5 lr5e-3 droptrain tokendrop0.01 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr5e-3_droptrain_tokendrop0.01_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 5 \
    --gs_lr 5e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_lambda_param_sqr 1e-1 \
    --seed 45

# bbh llama8b gs10 lr5e-3 droptrain tokendrop0.01 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr5e-3_droptrain_tokendrop0.01_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 5e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_lambda_param_sqr 1e-1 \
    --seed 45

# bbh llama8b gs15 lr5e-3 droptrain tokendrop0.01 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs15_lr5e-3_droptrain_tokendrop0.01_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 15 \
    --gs_lr 5e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_lambda_param_sqr 1e-1 \
    --seed 45

# bbh llama8b gs20 lr5e-3 droptrain tokendrop0.01 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr5e-3_droptrain_tokendrop0.01_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 5e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_lambda_param_sqr 1e-1 \
    --seed 45

# bbh llama8b gs25 lr5e-3 droptrain tokendrop0.01 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr5e-3_droptrain_tokendrop0.01_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 5e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_lambda_param_sqr 1e-1 \
    --seed 45

# 53.55995505546852
# 54.348677952034556
# 55.161965602311234
# 54.6999565676867
# 54.168559137402596
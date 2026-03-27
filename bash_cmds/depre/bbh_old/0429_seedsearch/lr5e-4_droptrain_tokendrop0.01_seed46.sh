# bbh llama8b gs5 lr5e-4 droptrain tokendrop0.01 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr5e-4_droptrain_tokendrop0.01_seed46 \
    --model_name llama8b \
    --gs_epochs 5 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --seed 46

# bbh llama8b gs10 lr5e-4 droptrain tokendrop0.01 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr5e-4_droptrain_tokendrop0.01_seed46 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --seed 46

# bbh llama8b gs15 lr5e-4 droptrain tokendrop0.01 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs15_lr5e-4_droptrain_tokendrop0.01_seed46 \
    --model_name llama8b \
    --gs_epochs 15 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --seed 46

# bbh llama8b gs20 lr5e-4 droptrain tokendrop0.01 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr5e-4_droptrain_tokendrop0.01_seed46 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --seed 46

# bbh llama8b gs25 lr5e-4 droptrain tokendrop0.01 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr5e-4_droptrain_tokendrop0.01_seed46 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --seed 46

# 50.060405204140665
# 52.76340291295426
# 53.12974436478924
# 53.58249985493257
# 53.55705370660507
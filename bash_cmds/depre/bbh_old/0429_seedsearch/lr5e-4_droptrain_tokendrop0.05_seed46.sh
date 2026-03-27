# bbh llama8b gs5 lr5e-4 droptrain tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr5e-4_droptrain_tokendrop0.05_seed46 \
    --model_name llama8b \
    --gs_epochs 5 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 46

# bbh llama8b gs10 lr5e-4 droptrain tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr5e-4_droptrain_tokendrop0.05_seed46 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 46

# bbh llama8b gs15 lr5e-4 droptrain tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs15_lr5e-4_droptrain_tokendrop0.05_seed46 \
    --model_name llama8b \
    --gs_epochs 15 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 46

# bbh llama8b gs20 lr5e-4 droptrain tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr5e-4_droptrain_tokendrop0.05_seed46 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 46

# bbh llama8b gs25 lr5e-4 droptrain tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr5e-4_droptrain_tokendrop0.05_seed46 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 5e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 46

# 49.774892694051886
# 53.1006297685427
# 53.16454516520985
# 52.98312074358934
# 53.895170748776586
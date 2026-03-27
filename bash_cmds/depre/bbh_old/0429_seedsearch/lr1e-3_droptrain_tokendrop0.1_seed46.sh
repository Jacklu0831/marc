# bbh llama8b gs5 lr1e-3 droptrain tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-3_droptrain_tokendrop0.1_seed46 \
    --model_name llama8b \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --seed 46

# bbh llama8b gs10 lr1e-3 droptrain tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_tokendrop0.1_seed46 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --seed 46

# bbh llama8b gs15 lr1e-3 droptrain tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs15_lr1e-3_droptrain_tokendrop0.1_seed46 \
    --model_name llama8b \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --seed 46

# bbh llama8b gs20 lr1e-3 droptrain tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_tokendrop0.1_seed46 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --seed 46

# bbh llama8b gs25 lr1e-3 droptrain tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-3_droptrain_tokendrop0.1_seed46 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --seed 46

# 52.73324207272362
# 52.73666214759602
# 53.95678704626861
# 53.10511367133169
# 54.12264089565517 <-
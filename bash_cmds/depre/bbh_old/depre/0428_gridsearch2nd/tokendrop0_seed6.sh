accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs2_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 2 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs4_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs6_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs8_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs12_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs14_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 14 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs16_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs18_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 18 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_tokendrop0_seed6 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 6

# 48.70764250458501
# 48.87851656670202
# 50.28542239243469
# 51.11024290444362
# 51.581237328578645
# 52.111346735806684
# 52.32268142661963
# 52.6665044548893
# 53.47779193284344
# 53.509155074459166 <-
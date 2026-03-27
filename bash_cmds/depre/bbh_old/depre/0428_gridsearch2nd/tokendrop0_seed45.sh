accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs2_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 2 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs4_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs6_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs8_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs12_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs14_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 14 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs16_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs18_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 18 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_tokendrop0_seed45 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 45

# 49.51184365773929
# 51.52990103762785
# 53.57409913117791
# 54.60147335770465
# 54.623862099767706
# 54.820125162431566
# 55.18408618940358
# 55.88049344031398
# 55.706627911679426
# 55.77791537204965 <-
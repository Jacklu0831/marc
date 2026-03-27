accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs2_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 2 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs4_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs6_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs8_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs12_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs14_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 14 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs16_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs18_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 18 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_tokendrop0_seed8 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 8

# 49.025436916762935
# 49.63840401195005
# 50.72587572379862
# 52.041096727454324
# 52.26573366320791
# 51.891453065846555
# 52.861850955071304
# 52.966605034983246
# 53.20314409505171
# 54.278944471699525 <-
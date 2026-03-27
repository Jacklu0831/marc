accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs2_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 2 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs4_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs6_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs8_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs12_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs14_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 14 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs16_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs18_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 18 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --gs_token_drop 0 \
    --seed 42

# 48.44090743642089
# 49.77587080029752
# 51.54836416675898
# 53.2600391066659
# 53.32579642026302
# 53.29280236890739
# 53.265782458620606
# 54.23952349301302
# 53.715007974313394
# 54.20672286491497 <-
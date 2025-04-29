accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs2_seed13 \
    --model_name llama8b \
    --gs_epochs 2 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs4_seed13 \
    --model_name llama8b \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs6_seed13 \
    --model_name llama8b \
    --gs_epochs 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs8_seed13 \
    --model_name llama8b \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_seed13 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs12_seed13 \
    --model_name llama8b \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs14_seed13 \
    --model_name llama8b \
    --gs_epochs 14 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs16_seed13 \
    --model_name llama8b \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs18_seed13 \
    --model_name llama8b \
    --gs_epochs 18 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_seed13 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout power \
    --seed 13

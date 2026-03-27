# bbh llama8b gs5 lr1e-3 droptrain tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-3_droptrain_tokendrop0.05_seed44 \
    --model_name llama8b \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 44

# bbh llama8b gs10 lr1e-3 droptrain tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_tokendrop0.05_seed44 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 44

# bbh llama8b gs15 lr1e-3 droptrain tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs15_lr1e-3_droptrain_tokendrop0.05_seed44 \
    --model_name llama8b \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 44

# bbh llama8b gs20 lr1e-3 droptrain tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_tokendrop0.05_seed44 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 44

# bbh llama8b gs25 lr1e-3 droptrain tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-3_droptrain_tokendrop0.05_seed44 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.05 \
    --seed 44

# 53.041459835660575
# 53.57145055134421
# 53.363709576737165
# 53.06713677310221
# 53.05152004304547
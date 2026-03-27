# bbh ttt llama8b gs2 lr1e-4 droptrain tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs2_lr1e-4_droptrain_tokendrop0.1_seed45 \
    --model_name llama8b \
    --gs_epochs 2 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs4 lr1e-4 droptrain tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs4_lr1e-4_droptrain_tokendrop0.1_seed45 \
    --model_name llama8b \
    --gs_epochs 4 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs6 lr1e-4 droptrain tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs6_lr1e-4_droptrain_tokendrop0.1_seed45 \
    --model_name llama8b \
    --gs_epochs 6 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs8 lr1e-4 droptrain tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs8_lr1e-4_droptrain_tokendrop0.1_seed45 \
    --model_name llama8b \
    --gs_epochs 8 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs10 lr1e-4 droptrain tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs10_lr1e-4_droptrain_tokendrop0.1_seed45 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs12 lr1e-4 droptrain tokendrop0.1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs12_lr1e-4_droptrain_tokendrop0.1_seed45 \
    --model_name llama8b \
    --gs_epochs 12 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.1 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# 55.35441295162132
# 55.27063870118042
# 55.24703666777446
# 55.36055414004899
# 55.37079897872519 <-
# 55.15773227056046
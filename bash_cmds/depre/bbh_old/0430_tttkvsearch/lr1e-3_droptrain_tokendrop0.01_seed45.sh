# bbh ttt llama8b gs2 lr1e-3 droptrain tokendrop0.01 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs2_lr1e-3_droptrain_tokendrop0.01_seed45 \
    --model_name llama8b \
    --gs_epochs 2 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs4 lr1e-3 droptrain tokendrop0.01 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs4_lr1e-3_droptrain_tokendrop0.01_seed45 \
    --model_name llama8b \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs6 lr1e-3 droptrain tokendrop0.01 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs6_lr1e-3_droptrain_tokendrop0.01_seed45 \
    --model_name llama8b \
    --gs_epochs 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs8 lr1e-3 droptrain tokendrop0.01 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs8_lr1e-3_droptrain_tokendrop0.01_seed45 \
    --model_name llama8b \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs10 lr1e-3 droptrain tokendrop0.01 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs10_lr1e-3_droptrain_tokendrop0.01_seed45 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# bbh ttt llama8b gs12 lr1e-3 droptrain tokendrop0.01 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs12_lr1e-3_droptrain_tokendrop0.01_seed45 \
    --model_name llama8b \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --seed 45

# 55.11177886094802 <-
# 54.73903685768093
# 54.75576577146866
# 54.881519462775685
# 54.68547839725971
# 54.66109827484038
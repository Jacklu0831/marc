# bbh ttt llama8b gs2 lr1e-3 droptrain tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs2_lr1e-3_droptrain_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 2 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run1_seed42 \
    --seed 42

# bbh ttt llama8b gs4 lr1e-3 droptrain tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs4_lr1e-3_droptrain_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run1_seed42 \
    --seed 42

# bbh ttt llama8b gs6 lr1e-3 droptrain tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs6_lr1e-3_droptrain_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run1_seed42 \
    --seed 42

# bbh ttt llama8b gs8 lr1e-3 droptrain tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs8_lr1e-3_droptrain_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run1_seed42 \
    --seed 42

# bbh ttt llama8b gs10 lr1e-3 droptrain tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs10_lr1e-3_droptrain_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run1_seed42 \
    --seed 42

# bbh ttt llama8b gs12 lr1e-3 droptrain tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_llama8b_gs12_lr1e-3_droptrain_tokendrop0_seed42 \
    --model_name llama8b \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --ttt_weight_dir eval_bbh_llama8b_ttt_iter8_save_run1_seed42 \
    --seed 42

# 55.887942433721776
# 55.94529902356424 <-
# 55.734742421764686
# 55.36549302709156
# 55.10928853650689
# 55.06852238698367
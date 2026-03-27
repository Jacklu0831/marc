# NOTE! USE 48GB!

# bbh ttt gs4 lr1e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs4_lr1e-3_tokendrop0.1_seed42 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed42 \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 42

# bbh ttt gs8 lr1e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs8_lr1e-3_tokendrop0.1_seed42 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed42 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 42

# bbh ttt gs12 lr1e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs12_lr1e-3_tokendrop0.1_seed42 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed42 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 42

# bbh ttt gs16 lr1e-3 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs16_lr1e-3_tokendrop0.1_seed42 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed42 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 42

# 58.32783607429226
# 58.01860886107019
# 57.59370909016528
# 57.86811180657829
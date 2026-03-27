# NOTE! USE 48GB!

# bbh ttt gs4 lr3e-4 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs4_lr3e-4_tokendrop0.1_seed42 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed42 \
    --gs_epochs 4 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 42

# bbh ttt gs8 lr3e-4 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs8_lr3e-4_tokendrop0.1_seed42 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed42 \
    --gs_epochs 8 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 42

# bbh ttt gs12 lr3e-4 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs12_lr3e-4_tokendrop0.1_seed42 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed42 \
    --gs_epochs 12 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 42

# bbh ttt gs16 lr3e-4 tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs16_lr3e-4_tokendrop0.1_seed42 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed42 \
    --gs_epochs 16 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 42

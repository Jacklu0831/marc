# NOTE! USE 48GB!

# bbh ttt gs4 lr1e-3 tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs4_lr1e-3_tokendrop0_seed46 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed46 \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0 \
    --seed 46

# bbh ttt gs8 lr1e-3 tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs8_lr1e-3_tokendrop0_seed46 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed46 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0 \
    --seed 46

# bbh ttt gs12 lr1e-3 tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs12_lr1e-3_tokendrop0_seed46 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed46 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0 \
    --seed 46

# bbh ttt gs16 lr1e-3 tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs16_lr1e-3_tokendrop0_seed46 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed46 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0 \
    --seed 46

# 58.06132979507591
# 58.21860103690599
# 58.33195888004324
# 57.856497862302824
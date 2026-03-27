# NOTE! USE 48GB!

# bbh ttt gs4 lr1e-4 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs4_lr1e-4_tokendrop0.05_seed43 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed43 \
    --gs_epochs 4 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 43

# bbh ttt gs8 lr1e-4 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs8_lr1e-4_tokendrop0.05_seed43 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed43 \
    --gs_epochs 8 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 43

# bbh ttt gs12 lr1e-4 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs12_lr1e-4_tokendrop0.05_seed43 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed43 \
    --gs_epochs 12 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 43

# bbh ttt gs16 lr1e-4 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs16_lr1e-4_tokendrop0.05_seed43 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed43 \
    --gs_epochs 16 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 43

# 56.42857845949624
# 56.48883172747011
# 56.545000028123695
# 56.43895083133655
# NOTE! USE 48GB!

# bbh ttt gs4 lr1e-3 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs4_lr1e-3_tokendrop0.05_seed43 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed43 \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 43

# bbh ttt gs8 lr1e-3 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs8_lr1e-3_tokendrop0.05_seed43 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed43 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 43

# bbh ttt gs12 lr1e-3 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs12_lr1e-3_tokendrop0.05_seed43 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed43 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 43

# bbh ttt gs16 lr1e-3 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs16_lr1e-3_tokendrop0.05_seed43 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed43 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 43

# 56.543395219533586
# 56.61648695622829
# 56.28850837523765
# 55.47732245508646
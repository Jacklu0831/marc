# NOTE! USE 48GB!

# bbh ttt gs4 lr3e-5 tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs4_lr3e-5_tokendrop0.05_seed46 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed46 \
    --gs_epochs 4 \
    --gs_lr 3e-5 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 46

# bbh ttt gs8 lr3e-5 tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs8_lr3e-5_tokendrop0.05_seed46 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed46 \
    --gs_epochs 8 \
    --gs_lr 3e-5 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 46

# bbh ttt gs12 lr3e-5 tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs12_lr3e-5_tokendrop0.05_seed46 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed46 \
    --gs_epochs 12 \
    --gs_lr 3e-5 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 46

# bbh ttt gs16 lr3e-5 tokendrop0.05 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs16_lr3e-5_tokendrop0.05_seed46 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed46 \
    --gs_epochs 16 \
    --gs_lr 3e-5 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.05 \
    --seed 46

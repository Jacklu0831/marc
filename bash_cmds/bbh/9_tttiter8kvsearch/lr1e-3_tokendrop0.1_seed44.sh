# NOTE! USE 48GB!

# bbh ttt gs4 lr1e-3 tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs4_lr1e-3_tokendrop0.1_seed44 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed44 \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 44

# bbh ttt gs8 lr1e-3 tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs8_lr1e-3_tokendrop0.1_seed44 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed44 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 44

# bbh ttt gs12 lr1e-3 tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs12_lr1e-3_tokendrop0.1_seed44 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed44 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 44

# bbh ttt gs16 lr1e-3 tokendrop0.1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_gs16_lr1e-3_tokendrop0.1_seed44 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run3_seed44 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0.1 \
    --seed 44

# 57.69354452307238
# 57.539774681802555
# 57.464640461447736
# 57.256007666224384
# bbh ttt iter8 save run3 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run3_seed42 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 42

# bbh ttt iter8 save run3 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run3_seed43 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 43

# bbh ttt iter8 save run3 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run3_seed44 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 44

# bbh ttt iter8 save run3 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run3_seed45 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 45

# bbh ttt iter8 save run3 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run3_seed46 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 46

# 57.478036532482406
# 56.587030896696014
# 56.64882180942553
# 58.1304043356473
# 58.455567534031154
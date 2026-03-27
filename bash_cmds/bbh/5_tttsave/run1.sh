# bbh ttt iter8 save run1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run1_seed42 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 42

# bbh ttt iter8 save run1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run1_seed43 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 43

# bbh ttt iter8 save run1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run1_seed44 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 44

# bbh ttt iter8 save run1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run1_seed45 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 45

# bbh ttt iter8 save run1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter8_save_run1_seed46 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --ttt_save \
    --seed 46

# 57.60700478645066
# 56.72591978558492
# 56.56244625042999
# 58.411734258920966
# 58.42259601208905

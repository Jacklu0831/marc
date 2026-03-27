# mmlu ttt gs2 lr1e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs2_lr1e-4_tokendrop0_seed45 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed45_run1 \
    --gs_epochs 2 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu ttt gs4 lr1e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs4_lr1e-4_tokendrop0_seed45 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed45_run1 \
    --gs_epochs 4 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu ttt gs6 lr1e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs6_lr1e-4_tokendrop0_seed45 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed45_run1 \
    --gs_epochs 6 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu ttt gs8 lr1e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs8_lr1e-4_tokendrop0_seed45 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed45_run1 \
    --gs_epochs 8 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu ttt gs10 lr1e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs10_lr1e-4_tokendrop0_seed45 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed45_run1 \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu ttt gs12 lr1e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs12_lr1e-4_tokendrop0_seed45 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed45_run1 \
    --gs_epochs 12 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu ttt gs14 lr1e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs14_lr1e-4_tokendrop0_seed45 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed45_run1 \
    --gs_epochs 14 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu ttt gs16 lr1e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs16_lr1e-4_tokendrop0_seed45 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed45_run1 \
    --gs_epochs 16 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 45

# 43.86137759491579
# 43.94761231944771
# 44.21440451838478
# 43.97912467880402
# 43.62263008955748
# 43.90975649093658
# 43.57879814599782
# 43.92219261267479
# mmlu ttt gs2 lr1e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs2_lr1e-4_tokendrop0.1_seed43 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed43_run1 \
    --gs_epochs 2 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --seed 43

# mmlu ttt gs4 lr1e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs4_lr1e-4_tokendrop0.1_seed43 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed43_run1 \
    --gs_epochs 4 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --seed 43

# mmlu ttt gs6 lr1e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs6_lr1e-4_tokendrop0.1_seed43 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed43_run1 \
    --gs_epochs 6 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --seed 43

# mmlu ttt gs8 lr1e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs8_lr1e-4_tokendrop0.1_seed43 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed43_run1 \
    --gs_epochs 8 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --seed 43

# mmlu ttt gs10 lr1e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs10_lr1e-4_tokendrop0.1_seed43 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed43_run1 \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --seed 43

# mmlu ttt gs12 lr1e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs12_lr1e-4_tokendrop0.1_seed43 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed43_run1 \
    --gs_epochs 12 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --seed 43

# mmlu ttt gs14 lr1e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs14_lr1e-4_tokendrop0.1_seed43 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed43_run1 \
    --gs_epochs 14 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --seed 43

# mmlu ttt gs16 lr1e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs16_lr1e-4_tokendrop0.1_seed43 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed43_run1 \
    --gs_epochs 16 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.1 \
    --seed 43

# 43.03317579625077
# 42.84125501699013
# 43.027621277159874
# 43.09846052091616
# 43.21159094208139
# 42.89240580384176
# 43.02298210203449
# 42.787217356478266
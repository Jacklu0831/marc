# mmlu ttt gs2 lr1e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs2_lr1e-4_tokendrop0_seed42 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed42_run1 \
    --gs_epochs 2 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu ttt gs4 lr1e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs4_lr1e-4_tokendrop0_seed42 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed42_run1 \
    --gs_epochs 4 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu ttt gs6 lr1e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs6_lr1e-4_tokendrop0_seed42 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed42_run1 \
    --gs_epochs 6 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu ttt gs8 lr1e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs8_lr1e-4_tokendrop0_seed42 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed42_run1 \
    --gs_epochs 8 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu ttt gs10 lr1e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs10_lr1e-4_tokendrop0_seed42 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed42_run1 \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu ttt gs12 lr1e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs12_lr1e-4_tokendrop0_seed42 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed42_run1 \
    --gs_epochs 12 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu ttt gs14 lr1e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs14_lr1e-4_tokendrop0_seed42 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed42_run1 \
    --gs_epochs 14 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu ttt gs16 lr1e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs16_lr1e-4_tokendrop0_seed42 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed42_run1 \
    --gs_epochs 16 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0 \
    --seed 42

# 43.911472837921885
# 43.57060068168709
# 43.53110213866531
# 43.365746010578
# 43.430851462607954
# 43.432139102707346
# 43.369170711616114
# 43.549514568496505
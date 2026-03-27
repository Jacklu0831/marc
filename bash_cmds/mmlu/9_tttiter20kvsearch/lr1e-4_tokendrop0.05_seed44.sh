# mmlu ttt gs2 lr1e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs2_lr1e-4_tokendrop0.05_seed44 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed44_run1 \
    --gs_epochs 2 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu ttt gs4 lr1e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs4_lr1e-4_tokendrop0.05_seed44 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed44_run1 \
    --gs_epochs 4 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu ttt gs6 lr1e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs6_lr1e-4_tokendrop0.05_seed44 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed44_run1 \
    --gs_epochs 6 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu ttt gs8 lr1e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs8_lr1e-4_tokendrop0.05_seed44 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed44_run1 \
    --gs_epochs 8 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu ttt gs10 lr1e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs10_lr1e-4_tokendrop0.05_seed44 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed44_run1 \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu ttt gs12 lr1e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs12_lr1e-4_tokendrop0.05_seed44 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed44_run1 \
    --gs_epochs 12 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu ttt gs14 lr1e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs14_lr1e-4_tokendrop0.05_seed44 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed44_run1 \
    --gs_epochs 14 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# mmlu ttt gs16 lr1e-4 tokendrop0.05 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_gs16_lr1e-4_tokendrop0.05_seed44 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed44_run1 \
    --gs_epochs 16 \
    --gs_lr 1e-4 \
    --gs_token_dropout 0.05 \
    --seed 44

# 43.32135506461444
# 42.9365610784935
# 43.35179817098453
# 43.3249657627421
# 43.093803687102955
# 43.08766800789671
# 43.41254520488684
# 43.199194414702745
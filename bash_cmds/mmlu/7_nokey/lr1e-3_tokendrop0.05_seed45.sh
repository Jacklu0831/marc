# mmlu gs5 lr1e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# mmlu gs10 lr1e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# mmlu gs15 lr1e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# mmlu gs20 lr1e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# mmlu gs25 lr1e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# 42.38017860630674
# 42.97894448980083
# 43.938509956489355
# 44.50495458034691
# 44.06863448099791
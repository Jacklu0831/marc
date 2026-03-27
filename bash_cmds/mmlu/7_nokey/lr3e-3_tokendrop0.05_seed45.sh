# mmlu gs5 lr3e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# mmlu gs10 lr3e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# mmlu gs15 lr3e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# mmlu gs20 lr3e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# mmlu gs25 lr3e-3 nokey tokendrop0.05 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_nokey_tokendrop0.05_seed45 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.05 \
    --gs_no_key \
    --seed 45

# 44.64717958411519
# 44.41971755455494
# 42.651718166585255
# 40.3097023816953
# 39.609348987102805
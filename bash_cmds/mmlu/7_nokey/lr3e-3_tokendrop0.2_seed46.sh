# mmlu gs5 lr3e-3 nokey tokendrop0.2 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_nokey_tokendrop0.2_seed46 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 46

# mmlu gs10 lr3e-3 nokey tokendrop0.2 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_nokey_tokendrop0.2_seed46 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 46

# mmlu gs15 lr3e-3 nokey tokendrop0.2 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_nokey_tokendrop0.2_seed46 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 46

# mmlu gs20 lr3e-3 nokey tokendrop0.2 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_nokey_tokendrop0.2_seed46 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 46

# mmlu gs25 lr3e-3 nokey tokendrop0.2 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_nokey_tokendrop0.2_seed46 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.2 \
    --gs_no_key \
    --seed 46

# 43.638158269990335
# 43.33982614275979
# 43.16109442325114
# 43.67935164772278
# 43.23505944117464
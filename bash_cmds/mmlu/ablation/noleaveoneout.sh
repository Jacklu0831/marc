# mmlu gs5 lr1e-3 dropnone tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_dropnone_tokendrop0.1_seed42 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 45

# mmlu gs10 lr1e-3 dropnone tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_dropnone_tokendrop0.1_seed42 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 45

# mmlu gs15 lr1e-3 dropnone tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_dropnone_tokendrop0.1_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 45

# mmlu gs20 lr1e-3 dropnone tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_dropnone_tokendrop0.1_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 45

# mmlu gs25 lr1e-3 dropnone tokendrop0.1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_dropnone_tokendrop0.1_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 45

# 41.397122538388295
# 41.57973764981921
# 42.45869139867192
# 42.40497551311277
# 43.01291175969823
# mmlu gs5 lr3e-2 tokendrop0.1 seed46 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-2_tokendrop0.1_seed46_ntoken32 \
    --gs_epochs 5 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 46

# mmlu gs10 lr3e-2 tokendrop0.1 seed46 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-2_tokendrop0.1_seed46_ntoken32 \
    --gs_epochs 10 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 46

# mmlu gs15 lr3e-2 tokendrop0.1 seed46 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-2_tokendrop0.1_seed46_ntoken32 \
    --gs_epochs 15 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 46

# mmlu gs20 lr3e-2 tokendrop0.1 seed46 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-2_tokendrop0.1_seed46_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 46

# mmlu gs25 lr3e-2 tokendrop0.1 seed46 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-2_tokendrop0.1_seed46_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 46

# 41.89495700955634
# 42.30104153110591
# 42.24978930846479
# 42.127235413763565
# 41.77774383962546
# mmlu gs5 lr3e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 5 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs10 lr3e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 10 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs15 lr3e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 15 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs20 lr3e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# mmlu gs25 lr3e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --gs_batch_size 4 \
    --seed 42

# 40.794836255194866
# 40.8694138913667
# 41.42621794677695
# 42.04053737519799
# 41.71029381735628
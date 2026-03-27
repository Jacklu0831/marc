# this finds that iter25 is best
# all under 2 hrs

# mmlu llama8b gs25 lr1e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_seed45 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# mmlu llama8b gs50 lr1e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs50_lr1e-3_randomkv_token_seed45 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# mmlu llama8b gs75 lr1e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs75_lr1e-3_randomkv_token_seed45 \
    --gs_epochs 75 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# mmlu llama8b gs100 lr1e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs100_lr1e-3_randomkv_token_seed45 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# 39.792836316620225 <-
# 37.316503411851926
# 36.4296647674488
# 35.97059649787578
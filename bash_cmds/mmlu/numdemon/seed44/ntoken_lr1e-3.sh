# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/mmlu/numdemon/seed44/ntoken_lr1e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# mmlu gs15 lr1e-3 randomkv token ntoken32 ndemon16 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_randomkv_token_ntoken32_ndemon16_seed44 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs20 lr1e-3 randomkv token ntoken32 ndemon16 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_randomkv_token_ntoken32_ndemon16_seed44 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs25 lr1e-3 randomkv token ntoken32 ndemon16 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_ntoken32_ndemon16_seed44 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs30 lr1e-3 randomkv token ntoken32 ndemon16 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_randomkv_token_ntoken32_ndemon16_seed44 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 64 \
    --seed 44








# mmlu gs15 lr1e-3 randomkv token ntoken32 ndemon32 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_randomkv_token_ntoken32_ndemon32_seed44 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs20 lr1e-3 randomkv token ntoken32 ndemon32 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_randomkv_token_ntoken32_ndemon32_seed44 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs25 lr1e-3 randomkv token ntoken32 ndemon32 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_ntoken32_ndemon32_seed44 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs30 lr1e-3 randomkv token ntoken32 ndemon32 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_randomkv_token_ntoken32_ndemon32_seed44 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 64 \
    --seed 44








# mmlu gs15 lr1e-3 randomkv token ntoken32 ndemon48 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_randomkv_token_ntoken32_ndemon48_seed44 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 48 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs20 lr1e-3 randomkv token ntoken32 ndemon48 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_randomkv_token_ntoken32_ndemon48_seed44 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 48 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs25 lr1e-3 randomkv token ntoken32 ndemon48 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_ntoken32_ndemon48_seed44 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 48 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs30 lr1e-3 randomkv token ntoken32 ndemon48 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_randomkv_token_ntoken32_ndemon48_seed44 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 48 \
    --filter_based_on_ndemo 64 \
    --seed 44








# mmlu gs15 lr1e-3 randomkv token ntoken32 ndemon64 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_randomkv_token_ntoken32_ndemon64_seed44 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 64 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs20 lr1e-3 randomkv token ntoken32 ndemon64 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_randomkv_token_ntoken32_ndemon64_seed44 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 64 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs25 lr1e-3 randomkv token ntoken32 ndemon64 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_ntoken32_ndemon64_seed44 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 64 \
    --filter_based_on_ndemo 64 \
    --seed 44

# mmlu gs30 lr1e-3 randomkv token ntoken32 ndemon64 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_randomkv_token_ntoken32_ndemon64_seed44 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 64 \
    --filter_based_on_ndemo 64 \
    --seed 44

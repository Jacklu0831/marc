# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/mmlu/numdemon/seed46/ctkv_lr1e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# mmlu gs15 lr1e-3 tokendrop0.1 ndemon16 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_ndemon16_seed46 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs20 lr1e-3 tokendrop0.1 ndemon16 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_ndemon16_seed46 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs25 lr1e-3 tokendrop0.1 ndemon16 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_ndemon16_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs30 lr1e-3 tokendrop0.1 ndemon16 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_tokendrop0.1_ndemon16_seed46 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 64 \
    --seed 46








# mmlu gs15 lr1e-3 tokendrop0.1 ndemon32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_ndemon32_seed46 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs20 lr1e-3 tokendrop0.1 ndemon32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_ndemon32_seed46 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs25 lr1e-3 tokendrop0.1 ndemon32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_ndemon32_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs30 lr1e-3 tokendrop0.1 ndemon32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_tokendrop0.1_ndemon32_seed46 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 64 \
    --seed 46












# mmlu gs15 lr1e-3 tokendrop0.1 ndemon48 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_ndemon48_seed46 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 48 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs20 lr1e-3 tokendrop0.1 ndemon48 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_ndemon48_seed46 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 48 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs25 lr1e-3 tokendrop0.1 ndemon48 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_ndemon48_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 48 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs30 lr1e-3 tokendrop0.1 ndemon48 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_tokendrop0.1_ndemon48_seed46 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 48 \
    --filter_based_on_ndemo 64 \
    --seed 46









# mmlu gs15 lr1e-3 tokendrop0.1 ndemon64 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_ndemon64_seed46 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 64 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs20 lr1e-3 tokendrop0.1 ndemon64 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_ndemon64_seed46 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 64 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs25 lr1e-3 tokendrop0.1 ndemon64 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_ndemon64_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 64 \
    --filter_based_on_ndemo 64 \
    --seed 46

# mmlu gs30 lr1e-3 tokendrop0.1 ndemon64 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_tokendrop0.1_ndemon64_seed46 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 64 \
    --filter_based_on_ndemo 64 \
    --seed 46

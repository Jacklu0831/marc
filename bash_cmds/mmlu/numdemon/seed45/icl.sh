# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/mmlu/numdemon/seed45/icl.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# mmlu normalicl seed45 ndemon16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_seed45_ndemon16 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 64 \
    --seed 45

# mmlu normalicl seed45 ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_seed45_ndemon32 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 64 \
    --seed 45

# mmlu normalicl seed45 ndemon48
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_seed45_ndemon48 \
    --num_demonstrations 48 \
    --filter_based_on_ndemo 64 \
    --seed 45

# mmlu normalicl seed45 ndemon64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_seed45_ndemon64 \
    --num_demonstrations 64 \
    --filter_based_on_ndemo 64 \
    --seed 45

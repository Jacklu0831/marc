# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_mmlu/pretrained/0401_2_gs_lr3e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs2 lr3e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs2_lr3e-3 \
    --gs_epochs 2 \
    --gs_lr 3e-3

# mmlu llama8b gs4 lr3e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs4_lr3e-3 \
    --gs_epochs 4 \
    --gs_lr 3e-3

# mmlu llama8b gs6 lr3e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs6_lr3e-3 \
    --gs_epochs 6 \
    --gs_lr 3e-3

# mmlu llama8b gs8 lr3e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs8_lr3e-3 \
    --gs_epochs 8 \
    --gs_lr 3e-3



# mmlu llama8b gs10 lr3e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3 \
    --gs_epochs 10 \
    --gs_lr 3e-3

# mmlu llama8b gs25 lr3e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3 \
    --gs_epochs 25 \
    --gs_lr 3e-3

# mmlu llama8b gs50 lr3e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs50_lr3e-3 \
    --gs_epochs 50 \
    --gs_lr 3e-3

# mmlu llama8b gs100 lr3e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs100_lr3e-3 \
    --gs_epochs 100 \
    --gs_lr 3e-3


# 2: 41.796964117254184
# 4: 41.10822423462288
# 6: 40.20098212382288
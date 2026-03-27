# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_mmlu/pretrained/0401_2_gs_lr1e-4.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs2 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs2_lr1e-4 \
    --gs_epochs 2 \
    --gs_lr 1e-4

# mmlu llama8b gs4 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs4_lr1e-4 \
    --gs_epochs 4 \
    --gs_lr 1e-4

# mmlu llama8b gs6 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs6_lr1e-4 \
    --gs_epochs 6 \
    --gs_lr 1e-4

# mmlu llama8b gs8 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs8_lr1e-4 \
    --gs_epochs 8 \
    --gs_lr 1e-4



# mmlu llama8b gs10 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-4 \
    --gs_epochs 10 \
    --gs_lr 1e-4

# mmlu llama8b gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-4 \
    --gs_epochs 25 \
    --gs_lr 1e-4

# mmlu llama8b gs50 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs50_lr1e-4 \
    --gs_epochs 50 \
    --gs_lr 1e-4

# mmlu llama8b gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs100_lr1e-4 \
    --gs_epochs 100 \
    --gs_lr 1e-4

# 2: 41.3560465804946
# 4: 41.30217833760541
# 6: 41.02170008502161
# 8: 40.98524790987165
# 10: 41.05126357054739
# 25: 41.58660874842066
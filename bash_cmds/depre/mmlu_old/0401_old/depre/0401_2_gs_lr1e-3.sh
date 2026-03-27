# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_mmlu/pretrained/0401_2_gs_lr1e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs2 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs2_lr1e-3 \
    --gs_epochs 2 \
    --gs_lr 1e-3

# mmlu llama8b gs4 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs4_lr1e-3 \
    --gs_epochs 4 \
    --gs_lr 1e-3

# mmlu llama8b gs6 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs6_lr1e-3 \
    --gs_epochs 6 \
    --gs_lr 1e-3

# mmlu llama8b gs8 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs8_lr1e-3 \
    --gs_epochs 8 \
    --gs_lr 1e-3



# mmlu llama8b gs10 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3 \
    --gs_epochs 10 \
    --gs_lr 1e-3

# mmlu llama8b gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3 \
    --gs_epochs 25 \
    --gs_lr 1e-3

# mmlu llama8b gs50 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs50_lr1e-3 \
    --gs_epochs 50 \
    --gs_lr 1e-3

# mmlu llama8b gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs100_lr1e-3 \
    --gs_epochs 100 \
    --gs_lr 1e-3




# 42.318046337640254
# 42.217533046115435
# 41.780424809513285
# 41.40244148942134

# 41.56393535470144
# 40.52060684758849
# 40.354269492964356
# 39.39487997009986
# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_mmlu/pretrained/0401_2_gs_droptrain.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs2 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs2_lr1e-3_droptrain \
    --gs_epochs 2 \
    --gs_lr 1e-3 \
    --gs_dropout train

# mmlu llama8b gs4 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs4_lr1e-3_droptrain \
    --gs_epochs 4 \
    --gs_lr 1e-3 \
    --gs_dropout train

# mmlu llama8b gs6 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs6_lr1e-3_droptrain \
    --gs_epochs 6 \
    --gs_lr 1e-3 \
    --gs_dropout train

# mmlu llama8b gs8 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs8_lr1e-3_droptrain \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_dropout train

# mmlu llama8b gs10 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_droptrain \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_dropout train

# mmlu llama8b gs25 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_droptrain \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train

















# mmlu llama8b gs2 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs2_lr1e-4_droptrain \
    --gs_epochs 2 \
    --gs_lr 1e-4 \
    --gs_dropout train

# mmlu llama8b gs4 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs4_lr1e-4_droptrain \
    --gs_epochs 4 \
    --gs_lr 1e-4 \
    --gs_dropout train

# mmlu llama8b gs6 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs6_lr1e-4_droptrain \
    --gs_epochs 6 \
    --gs_lr 1e-4 \
    --gs_dropout train

# mmlu llama8b gs8 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs8_lr1e-4_droptrain \
    --gs_epochs 8 \
    --gs_lr 1e-4 \
    --gs_dropout train

# mmlu llama8b gs10 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-4_droptrain \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_dropout train

# mmlu llama8b gs25 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-4_droptrain \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_dropout train


# 2: 42.061741409501344
# 4: 42.32337517011004
# 6: 43.019713845323736
# 8: 42.67564072808871
# 10: 43.237547402236416
# 25: 43.28969937167223 <-

# 2: 41.67337515090146
# 4: 41.38682153423949
# 6: 41.648752189990404
# 8: 41.6669618827898
# 10: 41.73922860624331
# 25: 42.65506536185904 <-

# 43.610794143690505
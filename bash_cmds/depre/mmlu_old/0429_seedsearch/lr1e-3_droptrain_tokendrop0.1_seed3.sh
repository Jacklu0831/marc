# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_mmlu/0429_seedsearch/lr1e-3_droptrain_tokendrop0.1_seed3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs5 lr1e-3 droptrain tokendrop0.1 seed3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_droptrain_tokendrop0.1_seed3 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --seed 3

# mmlu llama8b gs10 lr1e-3 droptrain tokendrop0.1 seed3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_droptrain_tokendrop0.1_seed3 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --seed 3

# mmlu llama8b gs15 lr1e-3 droptrain tokendrop0.1 seed3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_droptrain_tokendrop0.1_seed3 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --seed 3

# mmlu llama8b gs20 lr1e-3 droptrain tokendrop0.1 seed3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_droptrain_tokendrop0.1_seed3 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --seed 3

# mmlu llama8b gs25 lr1e-3 droptrain tokendrop0.1 seed3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_droptrain_tokendrop0.1_seed3 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --seed 3

# 42.064261516686265
# 41.576097397543954
# 41.90319902594785
# 41.8310023901112
# 42.96195650323542 <-
# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_mmlu/0401_prefixsearch/lr1e-2_droptrain_tokendrop0.01.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs10 lr1e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-2_droptrain_tokendrop0.01_ntoken32 \
    --gs_epochs 10 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_ntokens 32

# mmlu llama8b gs20 lr1e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-2_droptrain_tokendrop0.01_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_ntokens 32

# mmlu llama8b gs25 lr1e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-2_droptrain_tokendrop0.01_ntoken32 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_ntokens 32

# mmlu llama8b gs30 lr1e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-2_droptrain_tokendrop0.01_ntoken32 \
    --gs_epochs 30 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_ntokens 32

# mmlu llama8b gs40 lr1e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs40_lr1e-2_droptrain_tokendrop0.01_ntoken32 \
    --gs_epochs 40 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_ntokens 32

# mmlu llama8b gs50 lr1e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs50_lr1e-2_droptrain_tokendrop0.01_ntoken32 \
    --gs_epochs 50 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.01 \
    --gs_ntokens 32

# running
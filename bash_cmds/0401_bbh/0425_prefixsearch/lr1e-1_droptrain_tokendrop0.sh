# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_bbh/0425_prefixsearch/lr1e-1_droptrain_tokendrop0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)





# bbh llama8b gs10 lr1e-1 droptrain tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-1_droptrain_tokendrop0_ntoken32 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs20 lr1e-1 droptrain tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-1_droptrain_tokendrop0_ntoken32 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs25 lr1e-1 droptrain tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-1_droptrain_tokendrop0_ntoken32 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs30 lr1e-1 droptrain tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-1_droptrain_tokendrop0_ntoken32 \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs40 lr1e-1 droptrain tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-1_droptrain_tokendrop0_ntoken32 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs50 lr1e-1 droptrain tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-1_droptrain_tokendrop0_ntoken32 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0 \
    --gs_ntokens 32 \
    --seed 45

# 54.12771825616624
# 51.36893991746102
# 49.002327233467156
# 46.08111292225616
# 43.881213502350086
# 34.47908259348937
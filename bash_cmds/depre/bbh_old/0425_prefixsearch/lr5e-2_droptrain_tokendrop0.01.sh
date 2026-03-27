# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_bbh/0425_prefixsearch/lr5e-2_droptrain_tokendrop0.01.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)





# bbh llama8b gs10 lr5e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr5e-2_droptrain_tokendrop0.01_ntoken32 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 5e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs20 lr5e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr5e-2_droptrain_tokendrop0.01_ntoken32 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 5e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs25 lr5e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr5e-2_droptrain_tokendrop0.01_ntoken32 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 5e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs30 lr5e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr5e-2_droptrain_tokendrop0.01_ntoken32 \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 5e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs40 lr5e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr5e-2_droptrain_tokendrop0.01_ntoken32 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 5e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_ntokens 32 \
    --seed 45

# bbh llama8b gs50 lr5e-2 droptrain tokendrop0.01 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr5e-2_droptrain_tokendrop0.01_ntoken32 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 5e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_drop 0.01 \
    --gs_ntokens 32 \
    --seed 45

# 51.50050949444435
# 54.68795113776836
# 54.77928428119522 <-
# 53.00972523347065
# 50.919342941194046
# 50.148913928408774
# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --single --bash_files bash_cmds/nlp/6_prefixsearch_TOCHECK/lr3e-2_droptrain_tokendrop0.2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs25 lr3e-2 droptrain tokendrop0.2 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-2_droptrain_tokendrop0.2_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32

# nlp gs50 lr3e-2 droptrain tokendrop0.2 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr3e-2_droptrain_tokendrop0.2_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 3e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32

# nlp gs75 lr3e-2 droptrain tokendrop0.2 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs75_lr3e-2_droptrain_tokendrop0.2_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 75 \
    --gs_lr 3e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32

# nlp gs100 lr3e-2 droptrain tokendrop0.2 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-2_droptrain_tokendrop0.2_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --gs_ntokens 32

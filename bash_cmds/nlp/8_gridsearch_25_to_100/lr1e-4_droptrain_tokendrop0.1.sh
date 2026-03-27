# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --single --bash_files bash_cmds/nlp/8_gridsearch_25_to_100_TOCHECK/lr1e-4_droptrain_tokendrop0.1.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs50 lr1e-4 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-4_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1

# nlp gs100 lr1e-4 droptrain tokendrop0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-4_droptrain_tokendrop0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1

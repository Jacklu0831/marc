# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_prefixsearch/lr1e-1_droptrain.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs25 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-1_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-1 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs50 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-1_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-1 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs75 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs75_lr1e-1_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 75 \
    --gs_lr 1e-1 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-1_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-1 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# Submitted batch job 59761677

# 0.4194573332677502
# 0.4338509574821252
# 0.4426556795884358 <-
# 0.4362641506014218
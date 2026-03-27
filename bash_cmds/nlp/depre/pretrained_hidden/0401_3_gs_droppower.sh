# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained_hidden/0401_3_gs_droppower.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gshidden5 lr1e-2 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden5_lr1e-2_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden25 lr1e-2 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden25_lr1e-2_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden100 lr1e-2 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden100_lr1e-2_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden250 lr1e-2 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden250_lr1e-2_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-2 \
    --gs_dropout power \
    --eval_seeds 100











# nlp gshidden5 lr1e-3 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden5_lr1e-3_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden25 lr1e-3 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden25_lr1e-3_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden100 lr1e-3 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden100_lr1e-3_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden250 lr1e-3 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden250_lr1e-3_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --eval_seeds 100










# nlp gshidden5 lr1e-4 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden5_lr1e-4_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden25 lr1e-4 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden25_lr1e-4_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden100 lr1e-4 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden100_lr1e-4_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden250 lr1e-4 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden250_lr1e-4_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-4 \
    --gs_dropout power \
    --eval_seeds 100









# nlp gshidden5 lr1e-5 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden5_lr1e-5_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-5 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden25 lr1e-5 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden25_lr1e-5_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-5 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden100 lr1e-5 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden100_lr1e-5_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-5 \
    --gs_dropout power \
    --eval_seeds 100

# nlp gshidden250 lr1e-5 droppower
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden250_lr1e-5_droppower \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-5 \
    --gs_dropout power \
    --eval_seeds 100

# Submitted batch job 59708184
# Submitted batch job 59708185

# double check which job is for which lr

# lr1e-2
# 0.3784423097122801
# 0.37412101105251555
# 0.4099253290601604
# 0.4333808570183316 <-

# lr1e-3
# 0.3683770607353304
# 0.36602687680379437
# 0.3928640606011348
# 0.39846547996545 <-

# lr1e-4
# 0.3610671165473903
# 0.36272814552966987 <-
# 0.36129143491115867
# 0.36182564799315553
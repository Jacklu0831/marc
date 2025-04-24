# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained_hidden/0401_3_gs_droptrain.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gshidden5 lr1e-2 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden5_lr1e-2_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden25 lr1e-2 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden25_lr1e-2_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden100 lr1e-2 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden100_lr1e-2_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden250 lr1e-2 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden250_lr1e-2_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --eval_seeds 100











# nlp gshidden5 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden5_lr1e-3_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden25 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden25_lr1e-3_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden100 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden100_lr1e-3_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden250 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden250_lr1e-3_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --eval_seeds 100










# nlp gshidden5 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden5_lr1e-4_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden25 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden25_lr1e-4_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden100 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden100_lr1e-4_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden250 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden250_lr1e-4_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --eval_seeds 100









# nlp gshidden5 lr1e-5 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden5_lr1e-5_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-5 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden25 lr1e-5 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden25_lr1e-5_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-5 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden100 lr1e-5 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden100_lr1e-5_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-5 \
    --gs_dropout train \
    --eval_seeds 100

# nlp gshidden250 lr1e-5 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gshidden250_lr1e-5_droptrain \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-5 \
    --gs_dropout train \
    --eval_seeds 100

# Submitted batch job 59708201
# Submitted batch job 59708200

# double check which job is for which lr
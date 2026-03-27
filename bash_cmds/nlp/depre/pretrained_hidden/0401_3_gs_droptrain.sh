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


# Submitted batch job 59708201

# double check which job is for which lr

# lr1e-2
# 0.39508495607583394
# 0.4228519895341981
# 0.42518672190438106 <-
# 0.4221284117631187

# lr1e-3
# 0.3699450183688923
# 0.39784317881356807
# 0.42366694525752924 <-
# 0.4232455327910394

# lr1e-4
# 0.36107156968022613
# 0.36388662648359443
# 0.3941804776042477
# 0.41220747898280713 <-
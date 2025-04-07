# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_13_gs_weightdecay0.01.sh

# nlp gs5 lr1e-2 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs25 lr1e-2 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs100 lr1e-2 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs250 lr1e-2 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100




# nlp gs5 lr1e-3 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs25 lr1e-3 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs100 lr1e-3 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs250 lr1e-3 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100










# nlp gs5 lr1e-4 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs25 lr1e-4 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs100 lr1e-4 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs250 lr1e-4 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.01 \
    --eval_seeds 100


# weightdecay0.01

# lr1e-2
# Submitted batch job 59047652 # 0.400
# Submitted batch job 59047653 # 0.394
# Submitted batch job 59047654 # 0.390
# Submitted batch job 59047655 # 0.388

# lr1e-3
# Submitted batch job 59047656 # 0.371
# Submitted batch job 59047657 # 0.396
# Submitted batch job 59047658 # 0.401
# Submitted batch job 59047659 # 0.404

# lr1e-4
# Submitted batch job 59047660 # 0.361
# Submitted batch job 59047661 # 0.363
# Submitted batch job 59047662 # 0.374
# Submitted batch job 59047663 # 0.370
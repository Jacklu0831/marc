# python make_sbatch.py --ngpu 1 --time 4  --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_16_gs_lossoninput.sh

# nlp gs5 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp gs25 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp gs100 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp gs250 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_loss_on_input \
    --eval_seeds 100




# nlp gs5 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp gs25 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp gs100 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp gs250 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_loss_on_input \
    --eval_seeds 100










# nlp gs5 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp gs25 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp gs100 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp gs250 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_lossoninput \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_loss_on_input \
    --eval_seeds 100


# Submitted batch job 59111655 # 0.381
# Submitted batch job 59111656 # 0.389
# Submitted batch job 59111657 # 0.389
# Submitted batch job 59111658 # 0.390

# Submitted batch job 59111659 # 0.370
# Submitted batch job 59111660 # 0.386
# Submitted batch job 59111661 # 0.395
# Submitted batch job 59111662 # 0.405

# Submitted batch job 59111663 # 0.362
# Submitted batch job 59111664 # 0.362
# Submitted batch job 59111665 # 0.363
# Submitted batch job 59111666 # 0.373

# so far 0.405
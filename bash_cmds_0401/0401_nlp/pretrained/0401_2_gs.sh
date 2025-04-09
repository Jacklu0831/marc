# python make_sbatch.py --ngpu 1 --time 4  --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_2_gs.sh
# full batch gs, search iter and lr

# nlp gs5 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --eval_seeds 100

# nlp gs25 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --eval_seeds 100

# nlp gs100 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --eval_seeds 100

# nlp gs250 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --eval_seeds 100




# nlp gs5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --eval_seeds 100

# nlp gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --eval_seeds 100

# nlp gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --eval_seeds 100

# nlp gs250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --eval_seeds 100










# nlp gs5 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --eval_seeds 100

# nlp gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --eval_seeds 100

# nlp gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --eval_seeds 100

# nlp gs250 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --eval_seeds 100


# numparam 41103360, 3.3, 16.2, 66.2, 163.4

# lr1e-2
# Submitted batch job 59022384 # 0.400
# Submitted batch job 59022385 # 0.394
# Submitted batch job 59022386 # 0.390
# Submitted batch job 59022387 # 0.388

# lr1e-3
# Submitted batch job 59022388 # 0.371
# Submitted batch job 59034229 # 0.396
# Submitted batch job 59034230 # 0.401
# Submitted batch job 59034231 # 0.404

# lr1e-4
# Submitted batch job 59034232 # 0.361
# Submitted batch job 59034233 # 0.363
# Submitted batch job 59034234 # 0.374
# Submitted batch job 59034235 # 0.370

# so far 0.404
# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_5_gs_leaveoneout.sh

# nlp gs5 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --eval_seeds 100




# nlp gs5 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100










# nlp gs5 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100



# lr1e-2
# Submitted batch job 59237620 # 0.413
# Submitted batch job 59237621 # 0.424
# Submitted batch job 59237622 # 0.409
# Submitted batch job 59237623 # 0.406

# lr1e-3
# Submitted batch job 59237624 # 0.383
# Submitted batch job 59237625 # 0.409
# Submitted batch job 59237626 # 0.439
# Submitted batch job 59237627 # 0.437

# lr1e-4
# Submitted batch job 59237628 # 0.361
# Submitted batch job 59237629 # 0.362
# Submitted batch job 59237630 # 0.396
# Submitted batch job 59237631 # 0.414

# so far 0.439
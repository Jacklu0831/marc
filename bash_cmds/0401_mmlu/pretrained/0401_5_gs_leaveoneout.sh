# python make_sbatch.py --ngpu 1 --time 2 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_10_gs_leaveoneout.sh

# nlp gs5 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs5_lr1e-2_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs25_lr1e-2_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs100_lr1e-2_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs250_lr1e-2_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --eval_seeds 100




# nlp gs5 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs5_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs25_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs100_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs250_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100










# nlp gs5 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs5_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs25_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs100_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_0_leaveoneout/test_time_evaluate.py \
    --tag gs250_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100



# lr1e-2
# Submitted batch job 59161879 # 0.417
# Submitted batch job 59161880 # 0.424
# Submitted batch job 59161881 # 0.405
# Submitted batch job 59161882 # 0.411

# lr1e-3
# Submitted batch job 59161883 # 0.381
# Submitted batch job 59161884 # 0.411
# Submitted batch job 59161885 # 0.442
# Submitted batch job 59161886 # 0.435

# lr1e-4
# Submitted batch job 59161887 # 0.362
# Submitted batch job 59161888 # 0.363
# Submitted batch job 59161889 # 0.395
# Submitted batch job 59161890 # 0.414

# so far 0.442
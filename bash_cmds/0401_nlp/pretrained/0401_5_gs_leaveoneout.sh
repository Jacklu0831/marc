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
# Submitted batch job 59237620
# Submitted batch job 59237621
# Submitted batch job 59237622
# Submitted batch job 59237623

# lr1e-3
# Submitted batch job 59237624
# Submitted batch job 59237625
# Submitted batch job 59237626
# Submitted batch job 59237627

# lr1e-4
# Submitted batch job 59237628
# Submitted batch job 59237629
# Submitted batch job 59237630
# Submitted batch job 59237631
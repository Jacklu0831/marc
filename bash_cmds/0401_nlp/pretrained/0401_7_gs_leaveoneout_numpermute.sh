# python make_sbatch.py --ngpu 1 --time 6 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_7_gs_leaveoneout_numpermute.sh

# nlp gs5 lr1e-2 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs25 lr1e-2 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs100 lr1e-2 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs250 lr1e-2 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100




# nlp gs5 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs25 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs100 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs250 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100










# nlp gs5 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs25 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs100 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs250 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100



# lr1e-2
# Submitted batch job 59202062 # 0.374
# Submitted batch job 59202063 # 0.390
# Submitted batch job 59202064 # 0.393
# Submitted batch job 59202065 # 0.394

# lr1e-3
# Submitted batch job 59202066 # 0.356
# Submitted batch job 59202067 # 0.354
# Submitted batch job 59202068 # 0.407
# Submitted batch job 59202069 # 0.418

# lr1e-4
# Submitted batch job 59202070 # 0.345
# Submitted batch job 59202071 # 0.346
# Submitted batch job 59202072 # 0.354
# Submitted batch job 59202073

# terrible at 0.418
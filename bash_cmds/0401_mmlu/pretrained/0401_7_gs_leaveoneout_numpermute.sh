# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_7_gs_leaveoneout_numpermute.sh

# nlp gs5 lr1e-2 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs5_lr1e-2_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs25 lr1e-2 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs25_lr1e-2_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs100 lr1e-2 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs100_lr1e-2_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs250 lr1e-2 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs250_lr1e-2_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100




# nlp gs5 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs5_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs25 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs25_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs100 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs100_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs250 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs250_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100










# nlp gs5 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs5_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs25 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs25_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs100 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs100_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs250 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs250_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100



# lr1e-2
# Submitted batch job 59191500
# Submitted batch job 59191501
# Submitted batch job 59191502
# Submitted batch job 59191503

# lr1e-3
# Submitted batch job 59191504
# Submitted batch job 59191505
# Submitted batch job 59191506
# Submitted batch job 59191507

# lr1e-4
# Submitted batch job 59191508
# Submitted batch job 59191509
# Submitted batch job 59191510
# Submitted batch job 59191511
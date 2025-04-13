# python make_sbatch.py --ngpu 1 --time 6 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_7_gs_leaveoneout_numpermute.sh




# nlp ft gs5 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs25 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs100 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs250 lr1e-3 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100










# nlp ft gs5 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs25 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs100 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs250 lr1e-4 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100






# nlp ft gs5 lr1e-5 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-5_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs25 lr1e-5 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-5_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs100 lr1e-5 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-5_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs250 lr1e-5 leaveoneout permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-5_leaveoneout_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_leave_one_out \
    --gs_num_permute 1024 \
    --eval_seeds 100




# lr1e-3

# lr1e-4

# lr1e-5
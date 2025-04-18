# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_7_gs_ntokens_leaveoneout.sh




# nlp ft gs5 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-2_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs25 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-2_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs100 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-2_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs250 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-2_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100








# nlp ft gs5 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs25 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs100 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs250 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100







# nlp ft gs5 lr1e-4 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs25 lr1e-4 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs100 lr1e-4 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs250 lr1e-4 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_ntokens32_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100





# lr1e-2
# Submitted batch job 59237698 # 0.438
# Submitted batch job 59237699 # 0.437
# Submitted batch job 59237700 # 0.426
# Submitted batch job 59237701 # 0.432

# lr1e-3
# Submitted batch job 59237702 # 0.439
# Submitted batch job 59237703 # 0.439
# Submitted batch job 59237704 # 0.441
# Submitted batch job 59237705 # 0.438

# lr1e-4
# Submitted batch job 59237706 # 0.436
# Submitted batch job 59237707 # 0.438
# Submitted batch job 59237708 # 0.438
# Submitted batch job 59237709 # 0.438

# so far 0.441
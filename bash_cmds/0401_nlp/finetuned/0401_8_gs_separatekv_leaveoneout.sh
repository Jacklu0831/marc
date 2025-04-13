# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_8_gs_separatekv_leaveoneout.sh




# nlp ft gs5 lr1e-3 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs25 lr1e-3 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs100 lr1e-3 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs250 lr1e-3 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100









# nlp ft gs5 lr1e-4 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs25 lr1e-4 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs100 lr1e-4 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs250 lr1e-4 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100










# nlp ft gs5 lr1e-5 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-5_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs25 lr1e-5 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-5_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs100 lr1e-5 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-5_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ft gs250 lr1e-5 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-5_separatekv_leaveoneout \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100




# lr1e-3
# Submitted batch job 59237718
# Submitted batch job 59237719
# Submitted batch job 59237720
# Submitted batch job 59237721

# lr1e-4
# Submitted batch job 59237722
# Submitted batch job 59237723
# Submitted batch job 59237724
# Submitted batch job 59237725

# lr1e-5
# Submitted batch job 59237726
# Submitted batch job 59237727
# Submitted batch job 59237728
# Submitted batch job 59237729
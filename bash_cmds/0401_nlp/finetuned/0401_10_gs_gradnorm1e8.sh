# python make_sbatch.py --ngpu 1 --time 4  --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_10_gs_gradnorm1e8.sh
# full batch gs, search iter and lr

# ft nlp gs5 lr1e-2 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-2_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp gs25 lr1e-2 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-2_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp gs100 lr1e-2 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-2_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp gs250 lr1e-2 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-2_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100




# ft nlp gs5 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp gs25 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp gs100 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp gs250 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100










# ft nlp gs5 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp gs25 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp gs100 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp gs250 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# Submitted batch job 59035156
# Submitted batch job 59035157
# Submitted batch job 59035158
# Submitted batch job 59035159

# Submitted batch job 59035160
# Submitted batch job 59035161
# Submitted batch job 59035162
# Submitted batch job 59035163

# Submitted batch job 59035164
# Submitted batch job 59035165
# Submitted batch job 59035166
# Submitted batch job 59035167
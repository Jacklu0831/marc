# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_4_gs_beta0.9.sh
# full batch gs, search iter and lr

# ft nlp gs5 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-2_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs25 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-2_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs100 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-2_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs250 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-2_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100




# ft nlp gs5 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs25 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs100 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs250 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100










# ft nlp gs5 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs25 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs100 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs250 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100



# lr1e-2
# Submitted batch job 59046972
# Submitted batch job 59046973
# Submitted batch job 59046974
# Submitted batch job 59046975

# lr1e-3
# Submitted batch job 59046976
# Submitted batch job 59046977
# Submitted batch job 59046978
# Submitted batch job 59046979

# lr1e-4
# Submitted batch job 59046980
# Submitted batch job 59046981
# Submitted batch job 59046982
# Submitted batch job 59046983
# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_4_gs_beta0.9.sh
# full batch gs, search iter and lr

# ft nlp gs5 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-2 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs25 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-2 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs100 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-2 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs250 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-2 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100




# ft nlp gs5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100










# ft nlp gs5 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs250 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# Submitted batch job 59034486
# Submitted batch job 59034487
# Submitted batch job 59034488
# Submitted batch job 59034489

# Submitted batch job 59034490
# Submitted batch job 59034491
# Submitted batch job 59034492
# Submitted batch job 59034493

# Submitted batch job 59034494
# Submitted batch job 59034495
# Submitted batch job 59034496
# Submitted batch job 59034497
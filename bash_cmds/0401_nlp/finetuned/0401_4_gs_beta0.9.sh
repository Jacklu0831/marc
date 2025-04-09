# python make_sbatch.py --ngpu 1 --time 2 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_4_gs_beta0.9.sh
# full batch gs, search iter and lr





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






# ft nlp gs5 lr1e-5 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-5_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs25 lr1e-5 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-5_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs100 lr1e-5 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-5_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# ft nlp gs250 lr1e-5 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-5_beta0.9 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_beta2 0.9 \
    --eval_seeds 100



# lr1e-2
# Submitted batch job 59046972 # 0.388
# Submitted batch job 59046973 # 0.378
# Submitted batch job 59046974 # 0.378
# Submitted batch job 59046975 # 0.379

# lr1e-3
# Submitted batch job 59046976 # 0.432
# Submitted batch job 59046977 # 0.426
# Submitted batch job 59046978 # 0.418
# Submitted batch job 59046979 # 0.422

# lr1e-4
# Submitted batch job 59046980 # 0.440
# Submitted batch job 59046981 # 0.442
# Submitted batch job 59046982 # 0.439
# Submitted batch job 59046983 # 0.429

# lr1e-5
# Submitted batch job 59075603 # 0.440
# Submitted batch job 59075604 # 0.437
# Submitted batch job 59075605 # 0.441
# Submitted batch job 59075606 # 0.439

# compared to beta0.999, this gets the same performance not better
# interesting, beta0.9 is good for pretrained but not finetuned,
# reason maybe finetuned does not need that much further optimization, and beta0.9 is just
# for speeding up optimization

# so far 0.442




# AFTER PRECISION FIX

# lr1e-3
# Submitted batch job 59139509
# Submitted batch job 59139510
# Submitted batch job 59139511
# Submitted batch job 59139512

# lr1e-4
# Submitted batch job 59139513
# Submitted batch job 59139514
# Submitted batch job 59139515
# Submitted batch job 59139516

# lr1e-5
# Submitted batch job 59139517
# Submitted batch job 59139518
# Submitted batch job 59139519
# Submitted batch job 59139520
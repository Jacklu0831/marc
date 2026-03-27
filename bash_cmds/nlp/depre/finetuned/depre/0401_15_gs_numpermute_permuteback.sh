# python make_sbatch.py --ngpu 1 --time 5 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_15_gs_numpermute_permuteback.sh

# nlp ft gs5 lr1e-2 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-2_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp ft gs25 lr1e-2 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-2_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp ft gs100 lr1e-2 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-2_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp ft gs250 lr1e-2 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-2_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100





# nlp ft gs5 lr1e-3 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp ft gs25 lr1e-3 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp ft gs100 lr1e-3 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp ft gs250 lr1e-3 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100










# nlp ft gs5 lr1e-4 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp ft gs25 lr1e-4 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp ft gs100 lr1e-4 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp ft gs250 lr1e-4 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_permuten1024_permuteback \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100




# lr1e-2
# Submitted batch job 59084476 # 0.361
# Submitted batch job 59084477 # 0.369
# Submitted batch job 59084478 # 0.386
# Submitted batch job 59084479 # 0.378

# lr1e-3
# Submitted batch job 59084480 # 0.367
# Submitted batch job 59084481 # 0.358
# Submitted batch job 59084482 # 0.363
# Submitted batch job 59084483 # 0.376

# lr1e-4
# Submitted batch job 59084484 # 0.356
# Submitted batch job 59084485 # 0.366
# Submitted batch job 59084486 # 0.357
# Submitted batch job 59084487 # 0.360

# all plateau <0.4, dogshit
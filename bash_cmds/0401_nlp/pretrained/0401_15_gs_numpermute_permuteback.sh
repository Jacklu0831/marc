# python make_sbatch.py --ngpu 1 --time 5 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_15_gs_numpermute_permuteback.sh

# nlp gs5 lr1e-2 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp gs25 lr1e-2 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp gs100 lr1e-2 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp gs250 lr1e-2 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100




# nlp gs5 lr1e-3 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp gs25 lr1e-3 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp gs100 lr1e-3 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp gs250 lr1e-3 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100










# nlp gs5 lr1e-4 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp gs25 lr1e-4 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp gs100 lr1e-4 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100

# nlp gs250 lr1e-4 permuten1024 permuteback
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_permuten1024_permuteback \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_permute_back \
    --eval_seeds 100


# lr1e-2
# Submitted batch job 59084448
# Submitted batch job 59084449
# Submitted batch job 59084450
# Submitted batch job 59084451

# lr1e-3
# Submitted batch job 59084452
# Submitted batch job 59084453
# Submitted batch job 59084454
# Submitted batch job 59084455

# lr1e-4
# Submitted batch job 59084456
# Submitted batch job 59084457
# Submitted batch job 59084458
# Submitted batch job 59084459
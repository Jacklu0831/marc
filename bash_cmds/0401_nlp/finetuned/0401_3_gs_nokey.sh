# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_3_gs_nokey.sh
# full batch gs with NO KEY, search iter and lr

# ft nlp gs5 lr1e-2 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-2_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --eval_seeds 100 \
    --gs_no_key

# ft nlp gs25 lr1e-2 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-2_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --eval_seeds 100 \
    --gs_no_key

# ft nlp gs100 lr1e-2 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-2_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --eval_seeds 100 \
    --gs_no_key

# ft nlp gs250 lr1e-2 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-2_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --eval_seeds 100 \
    --gs_no_key




# ft nlp gs5 lr1e-3 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --eval_seeds 100 \
    --gs_no_key

# ft nlp gs25 lr1e-3 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --eval_seeds 100 \
    --gs_no_key

# ft nlp gs100 lr1e-3 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --eval_seeds 100 \
    --gs_no_key

# ft nlp gs250 lr1e-3 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --eval_seeds 100 \
    --gs_no_key










# ft nlp gs5 lr1e-4 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --eval_seeds 100 \
    --gs_no_key

# ft nlp gs25 lr1e-4 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --eval_seeds 100 \
    --gs_no_key

# ft nlp gs100 lr1e-4 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --eval_seeds 100 \
    --gs_no_key

# ft nlp gs250 lr1e-4 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_nokey \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --eval_seeds 100 \
    --gs_no_key


# Submitted batch job 59034472
# Submitted batch job 59034473
# Submitted batch job 59034474
# Submitted batch job 59034475

# Submitted batch job 59034476
# Submitted batch job 59034477
# Submitted batch job 59034478
# Submitted batch job 59034479

# Submitted batch job 59034480
# Submitted batch job 59034481
# Submitted batch job 59034482
# Submitted batch job 59034483
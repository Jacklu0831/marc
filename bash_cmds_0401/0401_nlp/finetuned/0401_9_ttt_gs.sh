# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_9_ttt_gs.sh

# ft nlp ttt250 permuten1000 gs5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs5_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs25_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs100_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs250_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100








# ft nlp ttt250 permuten1000 gs5 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs5_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs25_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs100_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs250 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs250_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100






# ft nlp ttt250 permuten1000 gs5 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs5_lr1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs25 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs25_lr1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs100 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs100_lr1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs250 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs250_lr1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100


# ttt permuten1000 iters250 gets 0.464
# we tune gs on top of it gslr1e-3 and 1e-4 and 1e-5
# result, goes up to 0.473

# gslr1e-3
# Submitted batch job 59034461 # 0.460
# Submitted batch job 59034462 # 0.467
# Submitted batch job 59034463 # 0.458
# Submitted batch job 59034464 # 0.464

# gslr1e-4
# Submitted batch job 59034465 # 0.461
# Submitted batch job 59034466 # 0.463
# Submitted batch job 59034467 # 0.463
# Submitted batch job 59034468 # 0.473

# gslr1e-5
# Submitted batch job 59099303 # 0.463
# Submitted batch job 59099304 # 0.460
# Submitted batch job 59099305 # 0.471
# Submitted batch job 59099306 # 0.468

# so far 0.473, again slightly boosting!
# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_11_ttt_gradnorm1e8.sh


# ft nlp ttt5 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt5_permuten1000_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 5 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp ttt25 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt25_permuten1000_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 25 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp ttt100 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt100_permuten1000_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 100 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# ft nlp ttt500 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt500_permuten1000_gradnorm1e8 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 500 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# Submitted batch job 59035172 # 0.394
# Submitted batch job 59035173 # 0.434
# Submitted batch job 59035174 # 0.458
# Submitted batch job 59035175 # 0.456
# Submitted batch job 59035176 # 0.458
# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_11_ttt_gradnorm1e8.sh


# nlp ttt5 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt5_permuten1000_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp ttt25 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt25_permuten1000_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp ttt100 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt100_permuten1000_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp ttt250 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt250_permuten1000_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp ttt500 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt500_permuten1000_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 1000 \
    --ttt_max_grad_norm 1e8 \
    --eval_seeds 100

# Submitted batch job 59035141
# Submitted batch job 59035142
# Submitted batch job 59035143
# Submitted batch job 59035144
# Submitted batch job 59035145
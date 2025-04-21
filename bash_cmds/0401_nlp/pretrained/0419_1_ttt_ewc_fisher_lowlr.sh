# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0419_1_ttt_ewc_fisher_lowlr.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# nlp ttt iter5 lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter5_lambda1e1_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e1 \
    --ttt_fisher \
    --eval_seeds 100

# nlp ttt iter25 lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter25_lambda1e1_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e1 \
    --ttt_fisher \
    --eval_seeds 100

# nlp ttt iter100 lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter100_lambda1e1_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e1 \
    --ttt_fisher \
    --eval_seeds 100

# nlp ttt iter250 lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter250_lambda1e1_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e1 \
    --ttt_fisher \
    --eval_seeds 100










# nlp ttt iter5 lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter5_lambda1e0_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e0 \
    --ttt_fisher \
    --eval_seeds 100

# nlp ttt iter25 lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter25_lambda1e0_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e0 \
    --ttt_fisher \
    --eval_seeds 100

# nlp ttt iter100 lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter100_lambda1e0_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e0 \
    --ttt_fisher \
    --eval_seeds 100

# nlp ttt iter250 lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter250_lambda1e0_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e0 \
    --ttt_fisher \
    --eval_seeds 100

# Submitted batch job 59510237
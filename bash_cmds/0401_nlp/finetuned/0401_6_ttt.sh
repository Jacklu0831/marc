# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_6_ttt.sh
# full batch ttt, use lr1e-4 and lora cfg from paper
# ttt paper bolded 1e-4 for ARC and BBH, no need to search further. 1e-2 seems too big so not trying it



# ft nlp ttt5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 5 \
    --eval_seeds 100

# ft nlp ttt25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt25 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 25 \
    --eval_seeds 100

# ft nlp ttt100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt100 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 100 \
    --eval_seeds 100

# ft nlp ttt250
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 250 \
    --eval_seeds 100

# ft nlp ttt500
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt500 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 500 \
    --eval_seeds 100








# nlp ft ttt5 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt5_permuten1000 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 5 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# nlp ft ttt25 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt25_permuten1000 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 25 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# nlp ft ttt100 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt100_permuten1000 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 100 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# nlp ft ttt250 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# nlp ft ttt500 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt500_permuten1000 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 500 \
    --ttt_permute_n 1000 \
    --eval_seeds 100


# permuten40
# Submitted batch job 59034438 # 0.418
# Submitted batch job 59034439 # 0.433
# Submitted batch job 59034440 # 0.449
# Submitted batch job 59034441 # 0.451
# Submitted batch job 59034442 # 0.462

# permuten1000
# Submitted batch job 59034426 # 0.389
# Submitted batch job 59034427 # 0.422
# Submitted batch job 59034428 # 0.455
# Submitted batch job 59034429 # 0.464
# Submitted batch job 59034430 # 0.452
# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_8_ttt_maxpermute.sh
# full batch ttt, use lr1e-4 and lora cfg from paper
# ttt paper bolded 1e-4 for ARC and BBH, no need to search further. 1e-2 seems too big so not trying it



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


# Submitted batch job 59034426
# Submitted batch job 59034427
# Submitted batch job 59034428
# Submitted batch job 59034429
# Submitted batch job 59034430
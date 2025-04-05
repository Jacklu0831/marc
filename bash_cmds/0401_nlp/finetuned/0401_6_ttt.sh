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


# Submitted batch job 59034438
# Submitted batch job 59034439
# Submitted batch job 59034440
# Submitted batch job 59034441
# Submitted batch job 59034442
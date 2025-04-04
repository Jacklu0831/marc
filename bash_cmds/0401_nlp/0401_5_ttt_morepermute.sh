# python make_sbatch.py --ngpu 1 --time 8 --bash_files bash_cmds/0401_nlp/0401_5_ttt_morepermute.sh --rtx8000
# full batch ttt, use lr1e-4 and lora cfg from paper
# ttt paper bolded 1e-4 for ARC and BBH, no need to search further. 1e-2 seems too big so not trying it



# nlp ttt5 permuten400
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt5_permuten400 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 400 \
    --eval_seeds 100

# nlp ttt25 permuten400
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt25_permuten400 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --ttt_permute_n 400 \
    --eval_seeds 100

# nlp ttt100 permuten400
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt100_permuten400 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --ttt_permute_n 400 \
    --eval_seeds 100

# nlp ttt250 permuten400
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt250_permuten400 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 400 \
    --eval_seeds 100

# Submitted batch job 59016370
# Submitted batch job 59016371
# Submitted batch job 59016372
# Submitted batch job 59016373
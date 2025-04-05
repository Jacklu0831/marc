# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_6_ttt.sh
# full batch ttt, use lr1e-4 and lora cfg from paper
# ttt paper bolded 1e-4 for ARC and BBH, no need to search further. 1e-2 seems too big so not trying it



# nlp ttt5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --eval_seeds 100

# nlp ttt25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt25 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --eval_seeds 100

# nlp ttt100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --eval_seeds 100

# nlp ttt250
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt250 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --eval_seeds 100

# nlp ttt500
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt500 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --eval_seeds 100


# numparams 47185920
# Submitted batch job 59029831 0.396
# Submitted batch job 59029832 0.414
# Submitted batch job 59029833 0.432
# Submitted batch job 59029834 0.432
# Submitted batch job 59029835
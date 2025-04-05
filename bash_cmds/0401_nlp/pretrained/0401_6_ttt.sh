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

# nlp ttt500 permuten400
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt500_permuten400 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 400 \
    --eval_seeds 100












# nlp ttt5 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt5_permuten1000 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# nlp ttt25 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt25_permuten1000 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# nlp ttt100 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt100_permuten1000 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# nlp ttt250 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt250_permuten1000 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# nlp ttt500 permuten1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt500_permuten1000 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 1000 \
    --eval_seeds 100



# numparams 47185920, times 9, 35, 134, 333sec

# permuten40
# Submitted batch job 59029831 # 0.396
# Submitted batch job 59029832 # 0.414
# Submitted batch job 59029833 # 0.432
# Submitted batch job 59029834 # 0.432
# Submitted batch job 59029835 # 0.441

# permuten400
# Submitted batch job 59016370 # 0.384
# Submitted batch job 59016371 # 0.409
# Submitted batch job 59016372 # 0.452
# Submitted batch job 59016373 # 0.456
# Submitted batch job 59025558 # 0.461

# permuten1000
# Submitted batch job 59025570 # 0.385
# Submitted batch job 59025571 # 0.404
# Submitted batch job 59025572 # 0.438
# Submitted batch job 59025573 # 0.449
# Submitted batch job 59025574 # 0.444

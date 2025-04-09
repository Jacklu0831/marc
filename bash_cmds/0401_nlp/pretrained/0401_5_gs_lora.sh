# python make_sbatch.py --ngpu 1 --time 2 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_5_gs_lora.sh

# nlp gs5 lr1e-3 lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100

# nlp gs25 lr1e-3 lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100

# nlp gs100 lr1e-3 lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100

# nlp gs250 lr1e-3 lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100









# nlp gs5 lr1e-3 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs25 lr1e-3 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs100 lr1e-3 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs250 lr1e-3 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100







# nlp gs5 lr1e-3 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_lora1e-5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# nlp gs25 lr1e-3 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_lora1e-5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# nlp gs100 lr1e-3 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_lora1e-5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# nlp gs250 lr1e-3 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_lora1e-5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100





# gslr1e-3
# numparams 88289280, same time gs without lora and almost no memory overhead

# gslora1e-3
# Submitted batch job 59034248 # 0.381
# Submitted batch job 59034249 # 0.416
# Submitted batch job 59034250 # 0.425
# Submitted batch job 59034251 # 0.417

# gslora1e-4
# Submitted batch job 59034252 # 0.393
# Submitted batch job 59034253 # 0.408
# Submitted batch job 59034254 # 0.412
# Submitted batch job 59034255 # 0.421

# gslora1e-5
# Submitted batch job 59034256 # 0.372
# Submitted batch job 59034257 # 0.398
# Submitted batch job 59034258 # 0.409
# Submitted batch job 59034259 # 0.406

# so far 0.425
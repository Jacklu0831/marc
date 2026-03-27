# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_14_gs_lora_beta0.9.sh

# nlp gs5 lr1e-3 beta0.9 lora1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs25 lr1e-3 beta0.9 lora1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs100 lr1e-3 beta0.9 lora1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs250 lr1e-3 beta0.9 lora1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100








# nlp gs5 lr1e-3 beta0.9 lora1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --gs_beta2 0.9 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs25 lr1e-3 beta0.9 lora1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --gs_beta2 0.9 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs100 lr1e-3 beta0.9 lora1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --gs_beta2 0.9 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs250 lr1e-3 beta0.9 lora1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --gs_beta2 0.9 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100


# gslora1e-3
# Submitted batch job 59075527 # 0.384
# Submitted batch job 59075528 # 0.418
# Submitted batch job 59075529 # 0.417
# Submitted batch job 59075530 # 0.405

# gslora1e-4
# Submitted batch job 59075531 # 0.397
# Submitted batch job 59075532 # 0.407
# Submitted batch job 59075533 # 0.412
# Submitted batch job 59075534 # 0.403

# so far 0.418
# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_5_gs_lora.sh

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






# Submitted batch job 59034248
# Submitted batch job 59034249
# Submitted batch job 59034250
# Submitted batch job 59034251

# Submitted batch job 59034252
# Submitted batch job 59034253
# Submitted batch job 59034254
# Submitted batch job 59034255

# Submitted batch job 59034256
# Submitted batch job 59034257
# Submitted batch job 59034258
# Submitted batch job 59034259
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


# gslora1e-3
# Submitted batch job 59139441 # 0.389
# Submitted batch job 59139442 # 0.383
# Submitted batch job 59139443 # 0.407
# Submitted batch job 59139444 # 0.427

# gslora1e-4
# Submitted batch job 59139445 # 0.397
# Submitted batch job 59139446 # 0.408
# Submitted batch job 59139447 # 0.409
# Submitted batch job 59139448 # 0.413

# gslora1e-5
# Submitted batch job 59139449 # 0.382
# Submitted batch job 59139450 # 0.399
# Submitted batch job 59139451 # 0.411
# Submitted batch job 59139452 # 0.412

# so far 0.427
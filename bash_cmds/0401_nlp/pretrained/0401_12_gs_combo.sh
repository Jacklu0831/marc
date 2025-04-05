# python make_sbatch.py --ngpu 1 --time 4  --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_12_gs_combo.sh

# gs should be 1e-3, 1e-4 barely changes score and 1e-2 is suboptimal
# search iter 50 100 200 300 400, 5 and 25 are not necessary because 100 always does better than them, search higher and more granular
# beta0.9 really improves performance, consider more experiments with lower beta1 and beta2
# lora1e-4 is good, 1e-3 is a bit worse and 1e-5 barely changes score
# try lorabeta0.9 and lorabeta0.999


# nlp gs50 lr1e-3 beta0.9 gslora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs50_lr1e-3_beta0.9_gslora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 50 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs100 lr1e-3 beta0.9 gslora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_beta0.9_gslora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs200 lr1e-3 beta0.9 gslora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs200_lr1e-3_beta0.9_gslora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 200 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs300 lr1e-3 beta0.9 gslora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs300_lr1e-3_beta0.9_gslora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 300 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs400 lr1e-3 beta0.9 gslora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs400_lr1e-3_beta0.9_gslora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 400 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100








# nlp gs50 lr1e-3 beta0.9 gslora1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs50_lr1e-3_beta0.9_gslora1e-4_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 50 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs100 lr1e-3 beta0.9 gslora1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_beta0.9_gslora1e-4_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs200 lr1e-3 beta0.9 gslora1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs200_lr1e-3_beta0.9_gslora1e-4_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 200 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs300 lr1e-3 beta0.9 gslora1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs300_lr1e-3_beta0.9_gslora1e-4_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 300 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100

# nlp gs400 lr1e-3 beta0.9 gslora1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs400_lr1e-3_beta0.9_gslora1e-4_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 400 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --gs_lora_beta2 0.9 \
    --eval_seeds 100


# Submitted batch job 59047743
# Submitted batch job 59047744
# Submitted batch job 59047745
# Submitted batch job 59047746
# Submitted batch job 59047747

# Submitted batch job 59047748
# Submitted batch job 59047749
# Submitted batch job 59047750
# Submitted batch job 59047751
# Submitted batch job 59047752
# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_5_gs_lora.sh

# ft nlp gs5 lr1e-3 lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_lora1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100

# ft nlp gs25 lr1e-3 lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_lora1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100

# ft nlp gs100 lr1e-3 lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_lora1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100

# ft nlp gs250 lr1e-3 lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_lora1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100









# ft nlp gs5 lr1e-3 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_lora1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# ft nlp gs25 lr1e-3 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_lora1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# ft nlp gs100 lr1e-3 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_lora1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# ft nlp gs250 lr1e-3 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_lora1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100







# ft nlp gs5 lr1e-3 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_lora1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# ft nlp gs25 lr1e-3 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_lora1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# ft nlp gs100 lr1e-3 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_lora1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# ft nlp gs250 lr1e-3 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_lora1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# gsiters250 gslr1e-3

# gslora1e-3
# Submitted batch job 59047167
# Submitted batch job 59047168
# Submitted batch job 59047169
# Submitted batch job 59047170

# gslora1e-4
# Submitted batch job 59047171
# Submitted batch job 59047172
# Submitted batch job 59047173
# Submitted batch job 59047174

# gslora1e-5
# Submitted batch job 59047175
# Submitted batch job 59047176
# Submitted batch job 59047177
# Submitted batch job 59047178
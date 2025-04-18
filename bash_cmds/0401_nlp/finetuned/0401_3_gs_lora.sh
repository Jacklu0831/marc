# python make_sbatch.py --ngpu 1 --time 2 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_5_gs_lora.sh



# ft nlp gs5 lr1e-4 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_lora1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# ft nlp gs25 lr1e-4 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_lora1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# ft nlp gs100 lr1e-4 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_lora1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# ft nlp gs250 lr1e-4 lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_lora1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100




# ft nlp gs5 lr1e-4 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_lora1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# ft nlp gs25 lr1e-4 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_lora1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# ft nlp gs100 lr1e-4 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_lora1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# ft nlp gs250 lr1e-4 lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_lora1e-5 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100






# ft nlp gs5 lr1e-4 lora1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_lora1e-6 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-6 \
    --eval_seeds 100

# ft nlp gs25 lr1e-4 lora1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_lora1e-6 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-6 \
    --eval_seeds 100

# ft nlp gs100 lr1e-4 lora1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_lora1e-6 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-6 \
    --eval_seeds 100

# ft nlp gs250 lr1e-4 lora1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_lora1e-6 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_lora \
    --gs_lora_lr 1e-6 \
    --eval_seeds 100

# lr1e-4
# Submitted batch job 59139523 # 0.439
# Submitted batch job 59139524 # 0.403
# Submitted batch job 59139525 # 0.404
# Submitted batch job 59139526 # 0.404

# lr1e-5
# Submitted batch job 59139527 # 0.441
# Submitted batch job 59139528 # 0.441
# Submitted batch job 59139529 # 0.436
# Submitted batch job 59139530 # 0.434

# lr1e-6
# Submitted batch job 59139531 # 0.441
# Submitted batch job 59139532 # 0.441
# Submitted batch job 59139533 # 0.434
# Submitted batch job 59139534 # 0.428

# so far 0.441
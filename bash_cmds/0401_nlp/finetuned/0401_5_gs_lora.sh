# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_5_gs_lora.sh

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




# lora1e-4
# Submitted batch job 59075581 # 0.436
# Submitted batch job 59075582 # 0.4
# Submitted batch job 59075583 # 0.4
# Submitted batch job 59075584 # 0.401

# lora1e-5
# Submitted batch job 59075585 # 0.443
# Submitted batch job 59075586 # 0.442
# Submitted batch job 59075587 # 0.433
# Submitted batch job 59075588 # 0.435

# lora1e-6
# Submitted batch job 59075589 # 0.440
# Submitted batch job 59075590 # 0.441
# Submitted batch job 59075591 # 0.436
# Submitted batch job 59075592 # 0.444

# so far 0.444
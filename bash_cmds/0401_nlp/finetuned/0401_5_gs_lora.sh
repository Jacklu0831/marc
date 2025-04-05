# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_5_gs_lora.sh

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

# Submitted batch job 59034446
# Submitted batch job 59034447
# Submitted batch job 59034448
# Submitted batch job 59034449

# Submitted batch job 59034450
# Submitted batch job 59034451
# Submitted batch job 59034452
# Submitted batch job 59034453

# Submitted batch job 59034454
# Submitted batch job 59034455
# Submitted batch job 59034456
# Submitted batch job 59034457
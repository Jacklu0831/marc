# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_13_gs_weightdecay0.03.sh

# ft nlp gs5 lr1e-2 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-2_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# ft nlp gs25 lr1e-2 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-2_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# ft nlp gs100 lr1e-2 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-2_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# ft nlp gs250 lr1e-2 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-2_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100




# ft nlp gs5 lr1e-3 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# ft nlp gs25 lr1e-3 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# ft nlp gs100 lr1e-3 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# ft nlp gs250 lr1e-3 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100










# ft nlp gs5 lr1e-4 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# ft nlp gs25 lr1e-4 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# ft nlp gs100 lr1e-4 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# ft nlp gs250 lr1e-4 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_wd0.03 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# weightdecay0.03

# lr1e-2
# Submitted batch job 59075421
# Submitted batch job 59075422
# Submitted batch job 59075423
# Submitted batch job 59075424

# lr1e-3
# Submitted batch job 59075425
# Submitted batch job 59075426
# Submitted batch job 59075427
# Submitted batch job 59075428

# lr1e-4
# Submitted batch job 59075429
# Submitted batch job 59075430
# Submitted batch job 59075431
# Submitted batch job 59075432
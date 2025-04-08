# python make_sbatch.py --ngpu 1 --time 4  --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_16_gs_lossoninput.sh

# nlp ft gs5 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-2_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs25 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-2_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs100 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-2_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs250 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-2_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_loss_on_input \
    --eval_seeds 100




# nlp ft gs5 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs25 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs100 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs250 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_loss_on_input \
    --eval_seeds 100










# nlp ft gs5 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs25 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs100 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs250 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_loss_on_input \
    --eval_seeds 100






# nlp ft gs5 lr1e-5 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-5_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs25 lr1e-5 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-5_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs100 lr1e-5 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-5_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_loss_on_input \
    --eval_seeds 100

# nlp ft gs250 lr1e-5 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-5_lossoninput \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_loss_on_input \
    --eval_seeds 100


# Submitted batch job 59111674
# Submitted batch job 59111675
# Submitted batch job 59111676
# Submitted batch job 59111677

# Submitted batch job 59111678
# Submitted batch job 59111679
# Submitted batch job 59111680
# Submitted batch job 59111681

# Submitted batch job 59111682
# Submitted batch job 59111683
# Submitted batch job 59111684
# Submitted batch job 59111685

# Submitted batch job 59111686
# Submitted batch job 59111687
# Submitted batch job 59111688
# Submitted batch job 59111689
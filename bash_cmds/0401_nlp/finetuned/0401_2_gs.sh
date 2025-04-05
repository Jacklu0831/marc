# python make_sbatch.py --ngpu 1 --time 4  --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_2_gs.sh
# full batch gs, search iter and lr

# nlp ft gs5 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-2 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --eval_seeds 100

# nlp ft gs25 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-2 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --eval_seeds 100

# nlp ft gs100 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-2 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --eval_seeds 100

# nlp ft gs250 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-2 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --eval_seeds 100




# nlp ft gs5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --eval_seeds 100

# nlp ft gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --eval_seeds 100

# nlp ft gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --eval_seeds 100

# nlp ft gs250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --eval_seeds 100










# nlp ft gs5 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs5_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --eval_seeds 100

# nlp ft gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs25_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --eval_seeds 100

# nlp ft gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs100_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --eval_seeds 100

# nlp ft gs250 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs250_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --eval_seeds 100

# Submitted batch job 59034410
# Submitted batch job 59034411
# Submitted batch job 59034412
# Submitted batch job 59034413

# Submitted batch job 59034414
# Submitted batch job 59034415
# Submitted batch job 59034416
# Submitted batch job 59034417

# Submitted batch job 59034418
# Submitted batch job 59034419
# Submitted batch job 59034420
# Submitted batch job 59034421
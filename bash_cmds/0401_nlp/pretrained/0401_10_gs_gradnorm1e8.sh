# python make_sbatch.py --ngpu 1 --time 4  --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_10_gs_gradnorm1e8.sh
# full batch gs, search iter and lr

# nlp gs5 lr1e-2 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp gs25 lr1e-2 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp gs100 lr1e-2 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp gs250 lr1e-2 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100




# nlp gs5 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp gs25 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp gs100 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp gs250 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100










# nlp gs5 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp gs25 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp gs100 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100

# nlp gs250 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_gradnorm1e8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8 \
    --eval_seeds 100


# gslr1e-2
# Submitted batch job 59035092 # 0.399
# Submitted batch job 59035093 # 0.396
# Submitted batch job 59035094 # 0.394
# Submitted batch job 59035095 # 0.388

# gslr1e-3
# Submitted batch job 59035096 # 0.371
# Submitted batch job 59035097 # 0.396
# Submitted batch job 59035098 # 0.401
# Submitted batch job 59035099 # 0.404

# gslr1e-4
# Submitted batch job 59035100 # 0.361
# Submitted batch job 59035101 # 0.363
# Submitted batch job 59035102 # 0.374
# Submitted batch job 59035103 # 0.370
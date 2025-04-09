# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_15_gs_numpermute.sh

# nlp gs5 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs25 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs100 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs250 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --eval_seeds 100




# nlp gs5 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs25 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs100 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs250 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --eval_seeds 100










# nlp gs5 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs25 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs100 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp gs250 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_permuten1024 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --eval_seeds 100


# lr1e-2
# Submitted batch job 59084463 # 0.369
# Submitted batch job 59084464 # 0.399
# Submitted batch job 59084465 # 0.401
# Submitted batch job 59084466 # 0.406

# lr1e-3
# Submitted batch job 59084467 # 0.345
# Submitted batch job 59084468 # 0.363
# Submitted batch job 59084469 # 0.412
# Submitted batch job 59084470 # 0.418

# lr1e-4
# Submitted batch job 59084471 # 0.345
# Submitted batch job 59084472 # 0.346
# Submitted batch job 59084473 # 0.353
# Submitted batch job 59084474 # 0.374

# so far 0.418, higher than normal gs
# benefit may be from obscuring memory to mitigate the retrieval issue,
# leading to higher generalization
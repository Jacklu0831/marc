# python make_sbatch.py --ngpu 1 --time 4  --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_16_gs_randominit.sh
# full batch gs, search iter and lr

# nlp gs5 lr1e-2 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_random_kv \
    --eval_seeds 100

# nlp gs25 lr1e-2 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_random_kv \
    --eval_seeds 100

# nlp gs100 lr1e-2 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_random_kv \
    --eval_seeds 100

# nlp gs250 lr1e-2 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_random_kv \
    --eval_seeds 100




# nlp gs5 lr1e-3 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_random_kv \
    --eval_seeds 100

# nlp gs25 lr1e-3 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_random_kv \
    --eval_seeds 100

# nlp gs100 lr1e-3 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_random_kv \
    --eval_seeds 100

# nlp gs250 lr1e-3 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_random_kv \
    --eval_seeds 100










# nlp gs5 lr1e-4 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_random_kv \
    --eval_seeds 100

# nlp gs25 lr1e-4 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_random_kv \
    --eval_seeds 100

# nlp gs100 lr1e-4 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_random_kv \
    --eval_seeds 100

# nlp gs250 lr1e-4 randominit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_randominit \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_random_kv \
    --eval_seeds 100


# Submitted batch job 59085560 # 0.346
# Submitted batch job 59085561 # 0.373
# Submitted batch job 59085562 # 0.387
# Submitted batch job 59085563 # 0.382

# Submitted batch job 59085564 # 0.329
# Submitted batch job 59085565 # 0.367
# Submitted batch job 59085566 # 0.380
# Submitted batch job 59085567 # 0.375

# Submitted batch job 59085568 # 0.332
# Submitted batch job 59085569 # 0.330
# Submitted batch job 59085570 # 0.337
# Submitted batch job 59085571 # 0.382

# so far dogshit (but this is good)
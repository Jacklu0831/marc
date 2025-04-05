# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_3_gs_nokey.sh
# full batch gs with NO KEY, search iter and lr

# nlp gs5 lr1e-2 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --eval_seeds 100 \
    --gs_no_key

# nlp gs25 lr1e-2 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --eval_seeds 100 \
    --gs_no_key

# nlp gs100 lr1e-2 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --eval_seeds 100 \
    --gs_no_key

# nlp gs250 lr1e-2 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --eval_seeds 100 \
    --gs_no_key




# nlp gs5 lr1e-3 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --eval_seeds 100 \
    --gs_no_key

# nlp gs25 lr1e-3 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --eval_seeds 100 \
    --gs_no_key

# nlp gs100 lr1e-3 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --eval_seeds 100 \
    --gs_no_key

# nlp gs250 lr1e-3 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --eval_seeds 100 \
    --gs_no_key










# nlp gs5 lr1e-4 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --eval_seeds 100 \
    --gs_no_key

# nlp gs25 lr1e-4 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --eval_seeds 100 \
    --gs_no_key

# nlp gs100 lr1e-4 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --eval_seeds 100 \
    --gs_no_key

# nlp gs250 lr1e-4 nokey
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_nokey \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --eval_seeds 100 \
    --gs_no_key

# numparam 20551680, times 3sec, 16sec, 65sec, 160sec
# lr1e-2
# Submitted batch job 59016380 # 0.399
# Submitted batch job 59016381 # 0.393
# Submitted batch job 59016382 # 0.383
# Submitted batch job 59016383 # 0.383

# lr1e-3
# Submitted batch job 59016384 # 0.370
# Submitted batch job 59016385 # 0.393
# Submitted batch job 59016386 # 0.401
# Submitted batch job 59016387 # 0.404

# lr1e-4
# Submitted batch job 59016388 # 0.361
# Submitted batch job 59016389 # 0.363
# Submitted batch job 59016390 # 0.373
# Submitted batch job 59016391 # 0.370
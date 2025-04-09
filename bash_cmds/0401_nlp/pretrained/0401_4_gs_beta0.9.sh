# python make_sbatch.py --ngpu 1 --time 2 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_4_gs_beta0.9.sh
# full batch gs, search iter and lr

# nlp gs5 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# nlp gs25 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# nlp gs100 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# nlp gs250 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --eval_seeds 100




# nlp gs5 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# nlp gs25 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# nlp gs100 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# nlp gs250 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --eval_seeds 100










# nlp gs5 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# nlp gs25 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# nlp gs100 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100

# nlp gs250 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_beta0.9 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --eval_seeds 100


# lr1e-2
# Submitted batch job 59032992 # 0.399
# Submitted batch job 59032993 # 0.391
# Submitted batch job 59032994 # 0.378
# Submitted batch job 59032995 # 0.377

# lr1e-3
# Submitted batch job 59032996 # 0.371
# Submitted batch job 59032997 # 0.393
# Submitted batch job 59032998 # 0.408
# Submitted batch job 59032999 # 0.416

# lr1e-4
# Submitted batch job 59033000 # 0.363
# Submitted batch job 59033001 # 0.363
# Submitted batch job 59033002 # 0.373
# Submitted batch job 59033003 # 0.374

# so far 0.416



# AFTER PRECISION FIX

# lr1e-2
# Submitted batch job 59139389
# Submitted batch job 59139390
# Submitted batch job 59139391
# Submitted batch job 59139392

# lr1e-3
# Submitted batch job 59139393
# Submitted batch job 59139394
# Submitted batch job 59139395
# Submitted batch job 59139396

# lr1e-4
# Submitted batch job 59139397
# Submitted batch job 59139398
# Submitted batch job 59139399
# Submitted batch job 59139400
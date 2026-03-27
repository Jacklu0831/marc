# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_13_gs_weightdecay0.03.sh

# nlp gs5 lr1e-2 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# nlp gs25 lr1e-2 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# nlp gs100 lr1e-2 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# nlp gs250 lr1e-2 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100




# nlp gs5 lr1e-3 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# nlp gs25 lr1e-3 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# nlp gs100 lr1e-3 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# nlp gs250 lr1e-3 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100










# nlp gs5 lr1e-4 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# nlp gs25 lr1e-4 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# nlp gs100 lr1e-4 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100

# nlp gs250 lr1e-4 wd0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_wd0.03 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.03 \
    --eval_seeds 100


# weightdecay0.03

# lr1e-2
# Submitted batch job 59075441 # 0.401
# Submitted batch job 59075442 # 0.392
# Submitted batch job 59075443 # 0.390
# Submitted batch job 59075444 # 0.388

# lr1e-3
# Submitted batch job 59075445 # 0.372
# Submitted batch job 59075446 # 0.396
# Submitted batch job 59075447 # 0.403
# Submitted batch job 59075448 # 0.404

# lr1e-4
# Submitted batch job 59075449 # 0.361
# Submitted batch job 59075450 # 0.363
# Submitted batch job 59075451 # 0.374
# Submitted batch job 59075452 # 0.371

# so far 0.404
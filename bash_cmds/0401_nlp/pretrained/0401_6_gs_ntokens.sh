# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_6_gs_ntokens.sh
# after max position fix

# nlp gs5 lr1e-1 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-1_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs25 lr1e-1 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-1_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-1 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-1_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr1e-1 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-1_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --eval_seeds 100









# nlp gs5 lr1e-2 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs25 lr1e-2 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-2 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr1e-2 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --eval_seeds 100








# nlp gs5 lr1e-3 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs25 lr1e-3 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-3 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr1e-3 ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_ntokens32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --eval_seeds 100




# lr1e-1
# Submitted batch job 59237632
# Submitted batch job 59237633
# Submitted batch job 59237634
# Submitted batch job 59237635

# lr1e-2
# Submitted batch job 59237636
# Submitted batch job 59237637
# Submitted batch job 59237638
# Submitted batch job 59237639

# lr1e-3
# Submitted batch job 59237640
# Submitted batch job 59237641
# Submitted batch job 59237642
# Submitted batch job 59237643
# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_7_gs_ntokens_leaveoneout.sh
# after max position fix

# nlp gs5 lr1e-1 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-1_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-1 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-1_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-1 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-1_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-1 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-1_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100









# nlp gs5 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100








# nlp gs5 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_ntokens32_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out \
    --eval_seeds 100





# lr1e-1
# Submitted batch job 59237650
# Submitted batch job 59237651
# Submitted batch job 59237652
# Submitted batch job 59237653

# lr1e-2
# Submitted batch job 59237654
# Submitted batch job 59237655
# Submitted batch job 59237656
# Submitted batch job 59237657

# lr1e-3
# Submitted batch job 59237658
# Submitted batch job 59237659
# Submitted batch job 59237660
# Submitted batch job 59237661
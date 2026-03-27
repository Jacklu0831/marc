# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0402_1_ttt_gs_leaveoneout.sh

# nlp ttt500 maxpermute nogs
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_nogs \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --eval_seeds 100





# nlp ttt500 maxpermute gs5 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs5_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ttt500 maxpermute gs25 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs25_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ttt500 maxpermute gs100 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs100_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ttt500 maxpermute gs250 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs250_lr1e-3_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --eval_seeds 100










# nlp ttt500 maxpermute gs5 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs5_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ttt500 maxpermute gs25 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs25_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ttt500 maxpermute gs100 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs100_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ttt500 maxpermute gs250 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs250_lr1e-4_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --eval_seeds 100





# nlp ttt500 maxpermute gs5 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs5_lr1e-5_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ttt500 maxpermute gs25 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs25_lr1e-5_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ttt500 maxpermute gs100 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs100_lr1e-5_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp ttt500 maxpermute gs250 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute_gs250_lr1e-5_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_leave_one_out \
    --eval_seeds 100


# Submitted batch job 59293440 # 0.413
# Submitted batch job 59293441 # 0.422
# Submitted batch job 59293442 # 0.406
# Submitted batch job 59293443 # 0.414

# Submitted batch job 59293444 # 0.382
# Submitted batch job 59293445 # 0.412
# Submitted batch job 59293446 # 0.437
# Submitted batch job 59293447 # 0.439

# Submitted batch job 59293448 # 0.362
# Submitted batch job 59293449 # 0.364
# Submitted batch job 59293450 # 0.396
# Submitted batch job 59293451 # 0.412

# wtf? rerun again





# Submitted batch job 59346128 # 0.460

# Submitted batch job 59346129
# Submitted batch job 59346130
# Submitted batch job 59346131
# Submitted batch job 59346132

# Submitted batch job 59346133
# Submitted batch job 59346134
# Submitted batch job 59346135
# Submitted batch job 59346136

# Submitted batch job 59346137
# Submitted batch job 59346138
# Submitted batch job 59346139
# Submitted batch job 59346140
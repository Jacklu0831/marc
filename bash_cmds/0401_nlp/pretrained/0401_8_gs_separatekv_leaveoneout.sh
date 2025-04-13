# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_8_gs_separatekv_leaveoneout.sh






# nlp gs5 lr1e-2 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-2 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-2 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-2 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100








# nlp gs5 lr1e-3 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-3 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-3 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-3 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100





# nlp gs5 lr1e-4 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs25 lr1e-4 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs100 lr1e-4 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100

# nlp gs250 lr1e-4 separatekv leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_separatekv_leaveoneout \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_separate_kv \
    --gs_leave_one_out \
    --eval_seeds 100





# lr1e-2
# Submitted batch job 59237666
# Submitted batch job 59237667
# Submitted batch job 59237668
# Submitted batch job 59237669

# lr1e-3
# Submitted batch job 59237670
# Submitted batch job 59237671
# Submitted batch job 59237672
# Submitted batch job 59237673

# lr1e-4
# Submitted batch job 59237674
# Submitted batch job 59237675
# Submitted batch job 59237676
# Submitted batch job 59237677
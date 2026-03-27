# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_8_gs_leaveoneout_separatekv.sh

# nlp gs5 lr1e-2 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-2_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100

# nlp gs25 lr1e-2 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-2_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100

# nlp gs100 lr1e-2 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-2_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100

# nlp gs250 lr1e-2 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-2_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100




# nlp gs5 lr1e-3 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-3_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100

# nlp gs25 lr1e-3 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-3_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100

# nlp gs100 lr1e-3 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-3_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100

# nlp gs250 lr1e-3 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-3_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100










# nlp gs5 lr1e-4 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs5_lr1e-4_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100

# nlp gs25 lr1e-4 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs25_lr1e-4_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100

# nlp gs100 lr1e-4 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs100_lr1e-4_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100

# nlp gs250 lr1e-4 leaveoneout separatekv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs250_lr1e-4_leaveoneout_separatekv \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_separate_kv \
    --eval_seeds 100




# lr1e-2
# Submitted batch job 59214164 # 0.354
# Submitted batch job 59214165 # 0.386
# Submitted batch job 59214166 # 0.392
# Submitted batch job 59214167 # 0.388

# lr1e-3
# Submitted batch job 59214168 # 0.321
# Submitted batch job 59214169 # 0.354
# Submitted batch job 59214170 # 0.394
# Submitted batch job 59214171 # 0.405

# lr1e-4
# Submitted batch job 59214172 # 0.314
# Submitted batch job 59214173 # 0.313
# Submitted batch job 59214174 # 0.326
# Submitted batch job 59214175 # 0.351

# terrible at 0.392
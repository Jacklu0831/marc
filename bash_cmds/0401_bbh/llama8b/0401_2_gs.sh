# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_cmds/0401_bbh/llama8b/0401_2_gs.sh

# bbh llama8b gs5 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-2 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs25 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-2 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs100 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-2 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs250 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-2 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5






# bbh llama8b gs5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-3 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-3 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-3 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-3 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5







# bbh llama8b gs5 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-4 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-4 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-4 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs250 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-4 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5






# bbh llama8b gs5 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-5 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs25 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-5 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs100 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-5 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs250 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-5 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5









# bbh llama8b gs5 lr1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-6 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs25 lr1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-6 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs100 lr1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-6 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs250 lr1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-6 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5





# lr1e-2
# Submitted batch job 59111739 # 30.36
# Submitted batch job 59111740 # 22.97
# Submitted batch job 59111741 # 20.61
# Submitted batch job 59111742 # 20.71

# lr1e-3
# Submitted batch job 59111743 # 48.87
# Submitted batch job 59111744 # 46.67
# Submitted batch job 59111745 # 46.68
# Submitted batch job 59111746 # 46.23

# lr1e-4
# Submitted batch job 59111747 # 49.24
# Submitted batch job 59111748 # 48.98
# Submitted batch job 59111749 # 49.32
# Submitted batch job 59111750 # 49.42

# so far 49.42




# AFTER PRECISION FIX and SCHEDULER FIX

# lr1e-3
# Submitted batch job 59139837
# Submitted batch job 59139838
# Submitted batch job 59139839
# Submitted batch job 59139840

# lr1e-4
# Submitted batch job 59139841
# Submitted batch job 59139842
# Submitted batch job 59139843
# Submitted batch job 59139844

# lr1e-5
# Submitted batch job 59139845
# Submitted batch job 59139846
# Submitted batch job 59139847
# Submitted batch job 59139848

# lr1e-6
# Submitted batch job 59139849
# Submitted batch job 59139850
# Submitted batch job 59139851
# Submitted batch job 59139852
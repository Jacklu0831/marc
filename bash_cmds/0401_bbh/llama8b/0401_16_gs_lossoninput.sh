# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_cmds/0401_bbh/llama8b/0401_16_gs_lossoninput.sh

# bbh llama8b gs5 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs5_lr1e-2_lossoninput \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs25 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs25_lr1e-2_lossoninput \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs100 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs100_lr1e-2_lossoninput \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs250 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs250_lr1e-2_lossoninput \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input






# bbh llama8b gs5 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs5_lr1e-3_lossoninput \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs25 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs25_lr1e-3_lossoninput \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs100 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs100_lr1e-3_lossoninput \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs250 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs250_lr1e-3_lossoninput \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input







# bbh llama8b gs5 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs5_lr1e-4_lossoninput \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs25 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs25_lr1e-4_lossoninput \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs100 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs100_lr1e-4_lossoninput \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs250 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs250_lr1e-4_lossoninput \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input




# bbh llama8b gs5 lr1e-5 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs5_lr1e-5_lossoninput \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs25 lr1e-5 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs25_lr1e-5_lossoninput \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs100 lr1e-5 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs100_lr1e-5_lossoninput \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs250 lr1e-5 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs250_lr1e-5_lossoninput \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input






# bbh llama8b gs5 lr1e-6 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs5_lr1e-6_lossoninput \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs25 lr1e-6 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs25_lr1e-6_lossoninput \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs100 lr1e-6 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs100_lr1e-6_lossoninput \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input

# bbh llama8b gs250 lr1e-6 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs250_lr1e-6_lossoninput \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_loss_on_input





# Submitted batch job 59112549 # 33.28
# Submitted batch job 59112550 # 28.49
# Submitted batch job 59112551 # 28.65
# Submitted batch job 59112552 # 28.38

# Submitted batch job 59112553 # 48.73
# Submitted batch job 59112554 # 46.69
# Submitted batch job 59112555 # 46.12
# Submitted batch job 59112556 # 46.61

# Submitted batch job 59112557 # 49.15
# Submitted batch job 59112558 # 48.84
# Submitted batch job 59112559 # 48.68
# Submitted batch job 59112560 # 48.83

# so far 49.15





# ABOVE HAS THE WRONG LR SCHEDULE

# lr1e-3
# Submitted batch job 59130918
# Submitted batch job 59130919
# Submitted batch job 59130920
# Submitted batch job 59130921

# lr1e-4
# Submitted batch job 59130922
# Submitted batch job 59130923
# Submitted batch job 59130924
# Submitted batch job 59130925

# lr1e-5
# Submitted batch job 59131345
# Submitted batch job 59131346
# Submitted batch job 59131347
# Submitted batch job 59131348

# lr1e-6
# Submitted batch job 59131349
# Submitted batch job 59131350
# Submitted batch job 59131351
# Submitted batch job 59131352
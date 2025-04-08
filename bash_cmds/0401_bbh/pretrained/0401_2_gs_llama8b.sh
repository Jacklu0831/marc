# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_cmds/0401_bbh/pretrained/0401_2_gs_llama8b.sh

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

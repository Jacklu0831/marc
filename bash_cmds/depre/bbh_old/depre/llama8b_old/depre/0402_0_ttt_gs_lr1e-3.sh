# python make_sbatch.py --ngpu 1 --time 1 --gb 64 --bash_files bash_cmds/0401_bbh/llama8b/0402_0_ttt_gs_lr1e-3.sh

# bbh llama8b ttt iter8 gs4 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_gs4_lr1e-3 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --gs_iters 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b ttt iter8 gs6 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_gs6_lr1e-3 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --gs_iters 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b ttt iter8 gs8 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_gs8_lr1e-3 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --gs_iters 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b ttt iter8 gs10 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_gs10_lr1e-3 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --gs_iters 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b ttt iter8 gs12 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_gs12_lr1e-3 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --gs_iters 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# fked up by enabling gradient checkpointing
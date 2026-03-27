# nah, run locally
# python make_sbatch.py --ngpu 1 --time 1 --gb 64 --bash_files bash_cmds/0401_bbh/llama8b/0402_0_ttt_maxgradnorm1e8.sh

# bbh llama8b ttt iter4 maxgradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter4_maxgradnorm1e8 \
    --model_name llama8b \
    --ttt_iters 4 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --ttt_max_grad_norm 1e8

# bbh llama8b ttt iter6 maxgradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter6_maxgradnorm1e8 \
    --model_name llama8b \
    --ttt_iters 6 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --ttt_max_grad_norm 1e8

# bbh llama8b ttt iter8 maxgradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_maxgradnorm1e8 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --ttt_max_grad_norm 1e8

# bbh llama8b ttt iter10 maxgradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter10_maxgradnorm1e8 \
    --model_name llama8b \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --ttt_max_grad_norm 1e8

# bbh llama8b ttt iter12 maxgradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter12_maxgradnorm1e8 \
    --model_name llama8b \
    --ttt_iters 12 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 60 \
    --ttt_max_grad_norm 1e8


# 52.10
# 52.69
# 53.28
# 53.72
# 54.15

# so far 54.15
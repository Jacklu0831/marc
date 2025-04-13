# nah, run locally
# python make_sbatch.py --ngpu 1 --time 1 --gb 64 --bash_files bash_cmds/0401_bbh/llama8b/0402_0_ttt.sh

# # bbh llama8b ttt iter4
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag bbh_llama8b_ttt_iter4 \
#     --model_name llama8b \
#     --ttt_iters 4 \
#     --ttt_gradient_checkpointing \
#     --ttt_permute_n 60

# # bbh llama8b ttt iter6
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag bbh_llama8b_ttt_iter6 \
#     --model_name llama8b \
#     --ttt_iters 6 \
#     --ttt_gradient_checkpointing \
#     --ttt_permute_n 60

# # bbh llama8b ttt iter8
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag bbh_llama8b_ttt_iter8 \
#     --model_name llama8b \
#     --ttt_iters 8 \
#     --ttt_gradient_checkpointing \
#     --ttt_permute_n 60

# # bbh llama8b ttt iter10
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag bbh_llama8b_ttt_iter10 \
#     --model_name llama8b \
#     --ttt_iters 10 \
#     --ttt_gradient_checkpointing \
#     --ttt_permute_n 60

# # bbh llama8b ttt iter12
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag bbh_llama8b_ttt_iter12 \
#     --model_name llama8b \
#     --ttt_iters 12 \
#     --ttt_gradient_checkpointing \
#     --ttt_permute_n 60

# # bbh llama8b ttt iter12
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag bbh_llama8b_ttt_iter12 \
#     --model_name llama8b \
#     --ttt_iters 12 \
#     --ttt_gradient_checkpointing \
#     --ttt_permute_n 60




# bbh llama8b ttt iter16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter16 \
    --model_name llama8b \
    --ttt_iters 16 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1000

# bbh llama8b ttt iter20
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter20 \
    --model_name llama8b \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1000

# bbh llama8b ttt iter24
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter24 \
    --model_name llama8b \
    --ttt_iters 24 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1000

# bbh llama8b ttt iter28
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter28 \
    --model_name llama8b \
    --ttt_iters 28 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1000


# iter4 52.52
# iter6 53.37
# iter8 53.82
# iter10 54.14
# iter12 54.80


# so far 54.80
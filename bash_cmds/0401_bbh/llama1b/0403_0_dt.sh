# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_bbh/llama1b/0403_0_dt.sh

# bbh llama7b demonstration5
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_llama7b/test_time_evaluate.py \
#     --tag test \
#     --model_name llama7b \
#     --num_demonstrations 5

# 34.00802397973718

# bbh llama7b demonstration5 dt5
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_llama7b/test_time_evaluate.py \
#     --tag test \
#     --model_name llama7b \
#     --num_demonstrations 5 \
#     --dt_iters 5

# 33.332506394391835

# bbh llama7b demonstration5 dt10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_llama7b/test_time_evaluate.py \
    --tag test \
    --model_name llama7b \
    --num_demonstrations 5 \
    --dt_iters 10

# 32.137632372770085

# bbh llama7b demonstration5 dt15
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_llama7b/test_time_evaluate.py \
    --tag test \
    --model_name llama7b \
    --num_demonstrations 5 \
    --dt_iters 15

# 31.092532026757887
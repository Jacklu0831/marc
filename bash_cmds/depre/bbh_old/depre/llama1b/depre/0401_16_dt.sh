# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_cmds/0401_bbh/llama1b/0401_16_dt.sh

# bbh llama1b dt5 lr1e-2 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_dt5_lr1e-2_demon5 \
    --model_name llama1b \
    --dt_iters 5 \
    --dt_lr 1e-2 \
    --num_demonstrations 5

# bbh llama1b dt10 lr1e-2 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_dt10_lr1e-2_demon5 \
    --model_name llama1b \
    --dt_iters 10 \
    --dt_lr 1e-2 \
    --num_demonstrations 5

# bbh llama1b dt15 lr1e-2 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_dt15_lr1e-2_demon5 \
    --model_name llama1b \
    --dt_iters 15 \
    --dt_lr 1e-2 \
    --num_demonstrations 5

# bbh llama1b dt20 lr1e-2 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_dt20_lr1e-2_demon5 \
    --model_name llama1b \
    --dt_iters 20 \
    --dt_lr 1e-2 \
    --num_demonstrations 5

# Submitted batch job 59163548 # 30.85
# Submitted batch job 59163549 # 30.70
# Submitted batch job 59163550 # 29.12
# Submitted batch job 59163551 # 26.06
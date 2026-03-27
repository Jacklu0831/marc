# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_bbh/llama1b/0401_6_ttt.sh

# bbh llama1b ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_ttt \
    --model_name llama1b \
    --ttt_iters 8 \
    --ttt_permute_n 40

# above uses 40 iters, below are 8 iters

# Submitted batch job 59190840 # 38.68
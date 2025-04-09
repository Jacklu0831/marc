# python make_sbatch.py --ngpu 1 --time 3 --bash_files bash_cmds/0401_bbh/llama1b/0401_6_ttt.sh

# bbh llama1b ttt onlylast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_ttt_onlylast \
    --model_name llama1b \
    --ttt_iters 40

# bbh llama1b ttt allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_ttt_allloss \
    --model_name llama1b \
    --ttt_iters 40 \
    --ttt_loss_type all

# Submitted batch job 59111717 # 37.31
# Submitted batch job 59111709 # 38.26
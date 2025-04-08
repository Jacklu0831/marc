# python make_sbatch.py --ngpu 1 --time 12 --bash_files bash_cmds/0401_bbh/llama8b/0401_6_ttt.sh

# bbh llama8b ttt onlylast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_onlylast \
    --model_name llama8b \
    --ttt_iters 40 \
    --ttt_gradient_checkpointing

# bbh llama8b ttt allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_allloss \
    --model_name llama8b \
    --ttt_iters 40 \
    --ttt_loss_type all \
    --ttt_gradient_checkpointing

# Submitted batch job 59111719
# Submitted batch job 59111710
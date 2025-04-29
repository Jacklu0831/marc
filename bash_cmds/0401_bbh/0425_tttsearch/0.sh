# python make_sbatch.py --ngpu 1 --time 3 --single --bash_files bash_cmds/0401_bbh/0425_tttsearch/0.sh
MASTER_PORT=$(comm -23 <(seq 10000 6125 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)









# bbh llama8b ttt iter5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter5 \
    --model_name llama8b \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --seed 45

# bbh llama8b ttt iter10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter10 \
    --model_name llama8b \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --seed 45

# bbh llama8b ttt iter15
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter15 \
    --model_name llama8b \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --seed 45

# bbh llama8b ttt iter20
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter20 \
    --model_name llama8b \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --seed 45

# bbh llama8b ttt iter25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter25 \
    --model_name llama8b \
    --ttt_iters 25 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --seed 45

# 54.832741634004506
# 55.21301395636723 <-
# 47.71317221175978
# 44.03993267112244

# had a lapse of judgement, theres no point of searching ttt when iter8 is best
# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_bbh/llama8b/0419_1_ttt_rslora.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)









# bbh llama8b ttt iter5 rslora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter5_rslora \
    --model_name llama8b \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --ttt_lora_rslora

# bbh llama8b ttt iter10 rslora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter10_rslora \
    --model_name llama8b \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --ttt_lora_rslora

# bbh llama8b ttt iter15 rslora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter15_rslora \
    --model_name llama8b \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --ttt_lora_rslora

# bbh llama8b ttt iter20 rslora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter20_rslora \
    --model_name llama8b \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --ttt_lora_rslora

# bbh llama8b ttt iter25 rslora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter25_rslora \
    --model_name llama8b \
    --ttt_iters 25 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --ttt_lora_rslora

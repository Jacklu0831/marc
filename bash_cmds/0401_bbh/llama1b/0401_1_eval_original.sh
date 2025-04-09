# run locally

# gs0 llama1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test_llama1b \
    --model_name llama1b

# 30.785162150233603

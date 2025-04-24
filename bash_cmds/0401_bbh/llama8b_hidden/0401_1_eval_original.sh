# run locally

accelerate launch --main_process_port $MASTER_PORT inference_bbh/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag test \
    --model_name llama1b \
    --eval_ratio 0.01 \
    --batch_size 2

# 38.888888888888886

accelerate launch --main_process_port $MASTER_PORT inference_bbh_hidden/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag test \
    --model_name llama1b \
    --eval_ratio 0.01 \
    --batch_size 2

# 38.888888888888886




accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --model_name llama1b \
    --eval_ratio 0.01 \
    --batch_size 2 \
    --gs_epochs 10 \
    --gs_batch_size 2

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag test \
    --model_name llama1b \
    --eval_ratio 0.01 \
    --batch_size 2 \
    --gs_epochs 10 \
    --gs_batch_size 2
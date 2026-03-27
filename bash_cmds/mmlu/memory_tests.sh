# works!
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test

# works!
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test_gs \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_batch_size 8

# works!
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test_ttt \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1000

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter10 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1000 \
    --ttt_batch_size 5 \
    --seed 42

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_iter50 \
    --ttt_iters 50 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1000 \
    --ttt_batch_size 5 \
    --seed 42

# prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_prompt_iter10 \
    --gs_epochs 10 \
    --ttt_gradient_checkpointing \
    --gs_batch_size 5 \
    --seed 42

# prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_prompt_iter50 \
    --gs_epochs 50 \
    --ttt_gradient_checkpointing \
    --gs_batch_size 5 \
    --seed 42

# 73.9690892000993
# 13.532959868510565
# 68.03168327609698
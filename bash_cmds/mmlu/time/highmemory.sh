# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_batch_size 4 \
    --seed 42

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter50 \
    --ttt_iters 50 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_batch_size 4 \
    --seed 42

# prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt_time/test_time_evaluate.py \
    --tag mmlu_prompt_iter10 \
    --gs_epochs 10 \
    --random_prompt token \
    --ttt_gradient_checkpointing \
    --gs_batch_size 4 \
    --seed 42

# prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt_time/test_time_evaluate.py \
    --tag mmlu_prompt_iter50 \
    --gs_epochs 50 \
    --random_prompt token \
    --ttt_gradient_checkpointing \
    --gs_batch_size 4 \
    --seed 42

# 7.65254188908471
# 25.961369262801277 -> 0.45772068434
# 4.386721173922221
# 21.994729889763725 -> 0.44020021789
# numbers too low, should be batch size 8
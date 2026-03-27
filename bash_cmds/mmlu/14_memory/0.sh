MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# prompt32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_prompt32_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_ratio 0.01 \
    --seed 42

# promptdemo
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_promptdemo_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_prompt token \
    --eval_ratio 0.01 \
    --seed 42

# prefix32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_prefix32_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --eval_ratio 0.01 \
    --seed 42

# prefixdemo
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_prefixdemo_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_kv token \
    --eval_ratio 0.01 \
    --seed 42

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_memory \
    --ttt_iters 1 \
    --ttt_batch_size 1 \
    --ttt_permute_n 1000 \
    --eval_ratio 0.01 \
    --seed 42

# ctprompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlu_ctprompt_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_ratio 0.01 \
    --seed 42

# ctkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ctkv_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_ratio 0.01 \
    --seed 42

# 13051.65853768808
# 18868.61131004051
# 13220.601236979166
# 15570.043158637152
# 18690.306396484375
# 18911.73260271991
# 15569.698757595486

# 10505.925925925925
# 841587.2962962963
# 14843.074074074075
# 103034.72222222222
# 714264.925925926
# 841587.2962962963
# 103034.72222222222

MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# prompt32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_prompt32_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_ratio 0.01 \
    --seed 42

# promptdemo
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_promptdemo_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_prompt token \
    --eval_ratio 0.01 \
    --seed 42

# prefix32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_prefix32_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --eval_ratio 0.01 \
    --seed 42

# prefixdemo
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_prefixdemo_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_kv token \
    --eval_ratio 0.01 \
    --seed 42

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ttt_memory \
    --ttt_iters 1 \
    --ttt_batch_size 1 \
    --ttt_permute_n 1000 \
    --eval_ratio 0.01 \
    --seed 42

# ctprompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbh_ctprompt_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_ratio 0.01 \
    --seed 42

# ctkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_ctkv_memory \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_ratio 0.01 \
    --seed 42

# 31640.567932128906
# 37493.264221191406
# 31549.7451171875
# 33292.51989746094
# 54752.15482584635
# 37571.05548095703
# 33292.523600260414

# 21736.833333333332
# 1276179.625
# 22264.166666666668
# 145405.875
# 1074490.1666666667
# 1276179.625
# 145405.875

MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# prompt32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt32_memory \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_ratio 0.01

# promptdemo
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_promptdemo_memory \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_prompt token \
    --eval_ratio 0.01

# prefix32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_prefix32_memory \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --eval_ratio 0.01

# prefixdemo
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_prefixdemo_memory \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout none \
    --random_kv token \
    --eval_ratio 0.01

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_memory \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 1 \
    --ttt_batch_size 1 \
    --ttt_permute_n 1000 \
    --eval_ratio 0.01

# ctprompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_ctprompt_memory \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_ratio 0.01

# ctkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ctkv_memory \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_ratio 0.01

# 3601.883956473214
# 7319.637351190477
# 3588.8245396205357
# 5648.132738095238
# 8883.869442894345
# 7318.7504510788685
# 5648.711593191964

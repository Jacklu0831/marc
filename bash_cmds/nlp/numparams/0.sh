# ctkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_nparam_ctkv_iter1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_batch_size 16 \
    --eval_ratio 0.01

# prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_nparam_prompt_iter1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_batch_size 4 \
    --eval_ratio 0.01

# 41634377.14285714
# 578255.2380952381
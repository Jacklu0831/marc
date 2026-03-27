# test saving ttt ckpts and loading + merging them




# ttt w high LR, save ckpts
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test_save_ttt \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 20 \
    --ttt_lr 1e-2 \
    --ttt_save \
    --eval_seeds 100 \
    --eval_ratio 0.01

# 0.3562576312576312

# load ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test_load_ttt \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_weight_dir eval_test_save_ttt_nlp_pretrained \
    --eval_seeds 100 \
    --eval_ratio 0.01

# 0.3562576312576312! goood
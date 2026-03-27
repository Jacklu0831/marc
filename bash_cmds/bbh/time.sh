# p-tuning 32token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbhtime3 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --seed 42 \
    --eval_ratio 0.01

# p-tuning demotoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbhtime4 \
    --gs_epochs 16 \
    --gs_lr 3e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --random_prompt token \
    --seed 42 \
    --eval_ratio 0.01

# kv-tuning 32token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbhtime5 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42 \
    --eval_ratio 0.01

# kv-tuning demotoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbhtime6 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 42 \
    --eval_ratio 0.01

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbhtime7 \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 40 \
    --seed 42 \
    --eval_ratio 0.01

# ct-p
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt/test_time_evaluate.py \
    --tag bbhtime1 \
    --gs_epochs 12 \
    --gs_lr 3e-4 \
    --gs_token_drop 0.1 \
    --ttt_gradient_checkpointing \
    --seed 42 \
    --eval_ratio 0.01

# ct-kv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbhtime2 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 42 \
    --eval_ratio 0.01

# tttkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbhtime8 \
    --ttt_weight_dir eval_bbh_ttt_iter8_save_run1_seed42 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_dropout 0 \
    --seed 42 \
    --eval_ratio 0.01

# 7.2193969289461775
# 35.094560692707695
# 6.837232947349548
# 8.424778014421463
# 12.965550442536673
# 26.304288645585377
# 8.43753664692243
# missing
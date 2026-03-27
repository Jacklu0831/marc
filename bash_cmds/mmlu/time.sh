# ct-p
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlutime1 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --ttt_gradient_checkpointing \
    --seed 42 \
    --eval_ratio 0.01

# ct-kv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlutime2 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --seed 42 \
    --eval_ratio 0.01

# p-tuning 32token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlutime3 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --ttt_gradient_checkpointing \
    --seed 42 \
    --eval_ratio 0.01

# p-tuning demotoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt/test_time_evaluate.py \
    --tag mmlutime4 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --ttt_gradient_checkpointing \
    --seed 43 \
    --eval_ratio 0.01

# kv-tuning 32token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlutime5 \
    --gs_epochs 24 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42 \
    --eval_ratio 0.01

# kv-tuning demotoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlutime6 \
    --gs_epochs 24 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 42 \
    --eval_ratio 0.01

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlutime7 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --seed 42 \
    --eval_ratio 0.01

# tttkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlutime8 \
    --ttt_weight_dir eval_mmlu_ttt_iter20_save_seed42_run1 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --seed 42 \
    --eval_ratio 0.01

# 18.35275075170729
# 3.3270759494216353
# 7.735388746968022
# 14.590619806890134
# 2.7333938987166793
# 4.014935965891238
# 10.670459027643558
# 1.6837743299978751

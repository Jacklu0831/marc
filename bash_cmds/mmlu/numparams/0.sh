# ctkv seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_nparam_ctkv_iter1_seed42 \
    --gs_epochs 1 \
    --gs_batch_size 8 \
    --seed 42 \
    --eval_ratio 0.01

# ctkv seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_nparam_ctkv_iter1_seed43 \
    --gs_epochs 1 \
    --gs_batch_size 8 \
    --seed 43 \
    --eval_ratio 0.01

# ctkv seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_nparam_ctkv_iter1_seed44 \
    --gs_epochs 1 \
    --gs_batch_size 8 \
    --seed 44 \
    --eval_ratio 0.01

# ctkv seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_nparam_ctkv_iter1_seed45 \
    --gs_epochs 1 \
    --gs_batch_size 8 \
    --seed 45 \
    --eval_ratio 0.01

# ctkv seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_nparam_ctkv_iter1_seed46 \
    --gs_epochs 1 \
    --gs_batch_size 8 \
    --seed 46 \
    --eval_ratio 0.01





# prompt seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt_time/test_time_evaluate.py \
    --tag mmlu_nparam_prompt_iter1_seed42 \
    --gs_epochs 1 \
    --random_prompt token \
    --ttt_gradient_checkpointing \
    --gs_batch_size 4 \
    --seed 42 \
    --eval_ratio 0.01

# prompt seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt_time/test_time_evaluate.py \
    --tag mmlu_nparam_prompt_iter1_seed43 \
    --gs_epochs 1 \
    --random_prompt token \
    --ttt_gradient_checkpointing \
    --gs_batch_size 4 \
    --seed 43 \
    --eval_ratio 0.01

# prompt seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt_time/test_time_evaluate.py \
    --tag mmlu_nparam_prompt_iter1_seed44 \
    --gs_epochs 1 \
    --random_prompt token \
    --ttt_gradient_checkpointing \
    --gs_batch_size 4 \
    --seed 44 \
    --eval_ratio 0.01

# prompt seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt_time/test_time_evaluate.py \
    --tag mmlu_nparam_prompt_iter1_seed45 \
    --gs_epochs 1 \
    --random_prompt token \
    --ttt_gradient_checkpointing \
    --gs_batch_size 4 \
    --seed 45 \
    --eval_ratio 0.01

# prompt seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt_time/test_time_evaluate.py \
    --tag mmlu_nparam_prompt_iter1_seed46 \
    --gs_epochs 1 \
    --random_prompt token \
    --ttt_gradient_checkpointing \
    --gs_batch_size 4 \
    --seed 46 \
    --eval_ratio 0.01

# ctkv
# 38987548.44444445
# 38787906.37037037
# 40630347.85185185
# 40709992.2962963
# 42517969.45454545
# avg: 2160361.7616162

# ctprompt
# 2088618.6666666667
# 2077923.5555555555
# 2176625.777777778
# 2180892.4444444445
# 2277748.3636363638
# avg 40326752.883502
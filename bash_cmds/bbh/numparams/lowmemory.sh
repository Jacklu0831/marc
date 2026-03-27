# ctkv seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_ctkv_iter1_seed42 \
    --gs_epochs 1 \
    --gs_batch_size 2 \
    --seed 42 \
    --eval_ratio 0.01

# ctkv seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_ctkv_iter1_seed43 \
    --gs_epochs 1 \
    --gs_batch_size 2 \
    --seed 43 \
    --eval_ratio 0.01

# ctkv seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_ctkv_iter1_seed44 \
    --gs_epochs 1 \
    --gs_batch_size 2 \
    --seed 44 \
    --eval_ratio 0.01

# ctkv seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_ctkv_iter1_seed45 \
    --gs_epochs 1 \
    --gs_batch_size 2 \
    --seed 45 \
    --eval_ratio 0.01

# ctkv seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_ctkv_iter1_seed46 \
    --gs_epochs 1 \
    --gs_batch_size 2 \
    --seed 46 \
    --eval_ratio 0.01







# prompt seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_prompt_iter1_seed42 \
    --gs_epochs 1 \
    --ttt_gradient_checkpointing \
    --gs_batch_size 5 \
    --seed 42 \
    --eval_ratio 0.01

# prompt seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_prompt_iter1_seed43 \
    --gs_epochs 1 \
    --ttt_gradient_checkpointing \
    --gs_batch_size 5 \
    --seed 43 \
    --eval_ratio 0.01

# prompt seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_prompt_iter1_seed44 \
    --gs_epochs 1 \
    --ttt_gradient_checkpointing \
    --gs_batch_size 5 \
    --seed 44 \
    --eval_ratio 0.01

# prompt seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_prompt_iter1_seed45 \
    --gs_epochs 1 \
    --ttt_gradient_checkpointing \
    --gs_batch_size 5 \
    --seed 45 \
    --eval_ratio 0.01

# prompt seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_prompt_iter1_seed46 \
    --gs_epochs 1 \
    --ttt_gradient_checkpointing \
    --gs_batch_size 5 \
    --seed 46 \
    --eval_ratio 0.01














# prefixm32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_prefixm32_iter1_seed42 \
    --gs_epochs 1 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 2 \
    --seed 42 \
    --eval_ratio 0.01

# prefixm32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_prefixm32_iter1_seed43 \
    --gs_epochs 1 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 2 \
    --seed 43 \
    --eval_ratio 0.01

# prefixm32 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_prefixm32_iter1_seed44 \
    --gs_epochs 1 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 2 \
    --seed 44 \
    --eval_ratio 0.01

# prefixm32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_prefixm32_iter1_seed45 \
    --gs_epochs 1 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 2 \
    --seed 45 \
    --eval_ratio 0.01

# prefixm32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_nparam_prefixm32_iter1_seed46 \
    --gs_epochs 1 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 2 \
    --seed 46 \
    --eval_ratio 0.01






# promptm32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_promptm32_iter1_seed42 \
    --gs_epochs 1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 2 \
    --seed 42 \
    --eval_ratio 0.01

# promptm32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_promptm32_iter1_seed43 \
    --gs_epochs 1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 2 \
    --seed 43 \
    --eval_ratio 0.01

# promptm32 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_promptm32_iter1_seed44 \
    --gs_epochs 1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 2 \
    --seed 44 \
    --eval_ratio 0.01

# promptm32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_promptm32_iter1_seed45 \
    --gs_epochs 1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 2 \
    --seed 45 \
    --eval_ratio 0.01

# promptm32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_nparam_promptm32_iter1_seed46 \
    --gs_epochs 1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 2 \
    --seed 46 \
    --eval_ratio 0.01

# ctkv
# 59299157.333333336
# 61095936.0
# 57181584.69565217
# 56990675.47826087
# 57936673.39130435

# ctprompt
# 3706197.3333333335
# 3818496.0
# 3573849.0434782607
# 3561917.217391304
# 3621042.086956522

# prefixm32
# 3683669.3333333335
# 3648170.6666666665
# 3670016.0
# 3670016.0
# 3670016.0

# promptm32
# 230229.33333333334
# 228010.66666666666
# 229376.0
# 229376.0
# 229376.0
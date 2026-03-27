# mmlu fisher seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_fisher/test_time_evaluate.py \
    --tag mmlu_fisher_seed42 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 42

# mmlu fisher seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_fisher/test_time_evaluate.py \
    --tag mmlu_fisher_seed43 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 43

# mmlu fisher seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_fisher/test_time_evaluate.py \
    --tag mmlu_fisher_seed44 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 44

# mmlu fisher seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_fisher/test_time_evaluate.py \
    --tag mmlu_fisher_seed45 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 45

# mmlu fisher seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_fisher/test_time_evaluate.py \
    --tag mmlu_fisher_seed46 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 46

# waiting to run this on a100, should be quick

# 'eval/gs_fisher_key': 2.6167823819040837e-08,
# 'eval/gs_fisher_val': 1.2502546057729374e-06,
# 'eval/gs_fisher_key': 2.9587441372303724e-08,
# 'eval/gs_fisher_val': 1.5374156715304263e-06,
# 'eval/gs_fisher_key': 2.868635360881564e-08,
# 'eval/gs_fisher_val': 1.4518719245825552e-06,
# 'eval/gs_fisher_key': 2.908770822238076e-08,
# 'eval/gs_fisher_val': 1.5210306108946902e-06,
# 'eval/gs_fisher_key': 2.6910143692026292e-08,
# 'eval/gs_fisher_val': 1.3259809981811731e-06,

# key: 2.8087894142913E-8
# val: 1.4173107621924E-6
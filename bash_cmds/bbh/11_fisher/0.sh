# bbh fisher seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_fisher/test_time_evaluate.py \
    --tag bbh_fisher_seed42 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 42

# bbh fisher seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_fisher/test_time_evaluate.py \
    --tag bbh_fisher_seed43 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 43

# bbh fisher seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_fisher/test_time_evaluate.py \
    --tag bbh_fisher_seed44 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 44

# bbh fisher seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_fisher/test_time_evaluate.py \
    --tag bbh_fisher_seed45 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 45

# bbh fisher seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_fisher/test_time_evaluate.py \
    --tag bbh_fisher_seed46 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001 \
    --seed 46

# waiting to run this on a100, should be quick

# 'eval/gs_fisher_key': 1.8065901718489208e-06,
# 'eval/gs_fisher_val': 0.00035616274582631985,
# 'eval/gs_fisher_key': 4.496421259750481e-06,
# 'eval/gs_fisher_val': 0.0010605555552496812,
# 'eval/gs_fisher_key': 1.0306826797348965e-06,
# 'eval/gs_fisher_val': 0.000186398576507036,
# 'eval/gs_fisher_key': 1.102610401976958e-06,
# 'eval/gs_fisher_val': 0.00020932968417899087,
# 'eval/gs_fisher_key': 1.02624040201136e-06,
# 'eval/gs_fisher_val': 0.00018373203651058455,


# key: 1.8925089830645E-6
# val: 0.00039923571965452
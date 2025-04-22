# run locally

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test \
    --eval_ratio 0.01

# 15mins
# {   'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/init_kv_time': 0.05631425087912041,
#     'eval/score': 41.21351337525021,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test

# 75mins
# {   'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/init_kv_time': 0.04006492464165939,
#     'eval/score': 42.21931786008163,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}
# run locally

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test \
    --eval_ratio 0.1 \
    --max_seq_len 4096

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
    --tag test \
    --eval_ratio 0.1 \
    --max_seq_len 2048

# {   'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/init_kv_time': 0.039571711891575864,
#     'eval/score': 41.16583984359498,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test \
    --eval_ratio 1.0 \
    --max_seq_len 4096

# 75mins
# {   'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/init_kv_time': 0.04006492464165939,
#     'eval/score': 42.21931786008163,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}





accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag test \
    --eval_ratio 0.1 \
    --max_seq_len 4096 \
    --delimiter space

# {   'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/init_kv_time': 0.03474988853722288,
#     'eval/score': 39.90835652525293,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}
# run locally


# # gs0 llama8b seed0
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag test \
#     --model_name llama8b \
#     --seed 0

# # 48.937178763533026




# # gs0 llama8b seed42
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag test \
#     --model_name llama8b \
#     --seed 42

# 47.10017654268236

# # gs0 llama8b seed43
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag test \
#     --model_name llama8b \
#     --seed 43

# 48.81980821204816

# # gs0 llama8b seed44
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag test \
#     --model_name llama8b \
#     --seed 44

# 48.626277252897395

# # gs0 llama8b seed45
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag test \
#     --model_name llama8b \
#     --seed 45

# 48.4337463799079

# # gs0 llama8b seed46
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
#     --tag test \
#     --model_name llama8b \
#     --seed 46

# 47.5285782863051


# # gs0 llama8b (64gb memory)
# accelerate launch --main_process_port $MASTER_PORT inference_bbh_debug/test_time_evaluate.py \
#     --untrainable_nbit 32 \
#     --tag test \
#     --model_name llama8b \
#     --seed 42 \
#     --batch_size 2

# # 46.955284938834296????? lmao even worse
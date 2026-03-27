# ctkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ctkv_iter10 \
    --gs_epochs 10 \
    --gs_batch_size 8 \
    --seed 42

# ctkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ctkv_iter50 \
    --gs_epochs 50 \
    --gs_batch_size 8 \
    --seed 42

# prefixm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_prefixm32_iter10 \
    --gs_epochs 10 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 8 \
    --seed 42

# prefixm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_prefixm32_iter50 \
    --gs_epochs 50 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 8 \
    --seed 42

# promptm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt_time/test_time_evaluate.py \
    --tag mmlu_promptm32_iter10 \
    --gs_epochs 10 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 8 \
    --seed 42

# promptm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_prompt_time/test_time_evaluate.py \
    --tag mmlu_promptm32_iter50 \
    --gs_epochs 50 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 8 \
    --seed 42

# 2.6974479224946766
# 13.432636980657223 -> 0.26837972645
# 1.8108449732815777
# 8.976198395093283 -> 0.17913383554
# 1.0006318931226377
# 4.911780282303139 -> 0.09777870972 -> should be about double this
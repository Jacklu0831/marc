# ct-kv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_time_ctkv_iter10 \
    --gs_epochs 10 \
    --gs_batch_size 2 \
    --seed 42

# ct-kv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_time_ctkv_iter50 \
    --gs_epochs 50 \
    --gs_batch_size 2 \
    --seed 42

# prefixm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_time_prefixm32_iter10 \
    --gs_epochs 10 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 2 \
    --seed 42

# prefixm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_time_prefixm32_iter50 \
    --gs_epochs 50 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 2 \
    --seed 42

# promptm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_time_promptm32_iter10 \
    --gs_epochs 10 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 2 \
    --seed 42

# promptm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_time_promptm32_iter50 \
    --gs_epochs 50 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 2 \
    --seed 42

# 5.2903592983881635
# 27.040072908004124
# 4.493413746356964
# 21.9238819082578
# 0.9321695963541666
# 4.466688613096873 # should be about 6-8 times this
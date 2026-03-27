# ctkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ctkv_iter10 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 10 \
    --gs_batch_size 16 \
    --eval_seeds 100

# ctkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ctkv_iter50 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_batch_size 16 \
    --eval_seeds 100

# prefixm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_prefixm32_iter10 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 10 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 16 \
    --eval_seeds 100

# prefixm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_prefixm32_iter50 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 16 \
    --eval_seeds 100

# promptm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_promptm32_iter10 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 10 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 16 \
    --eval_seeds 100

# promptm32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_promptm32_iter50 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 16 \
    --eval_seeds 100

# 4.936392046156383
# 25.202397323790052
# 4.297981398446219
# 21.866185767310007
# 6.793002071834746
# 35.028306086858116
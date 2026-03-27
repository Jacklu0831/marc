# bbh prompt8 lr3e-4 random uniform seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_prompt8_lr3e-4_random_uniform_seed44 \
    --gs_epochs 8 \
    --gs_lr 3e-4 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --gs_grad_accum_steps 8 \
    --seed 44

# bbh prompt12 lr3e-4 random uniform seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_prompt12_lr3e-4_random_uniform_seed44 \
    --gs_epochs 12 \
    --gs_lr 3e-4 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --gs_grad_accum_steps 8 \
    --seed 44

# bbh prompt16 lr3e-4 random uniform seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_prompt16_lr3e-4_random_uniform_seed44 \
    --gs_epochs 16 \
    --gs_lr 3e-4 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --gs_grad_accum_steps 8 \
    --seed 44

# bbh prompt20 lr3e-4 random uniform seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_prompt20_lr3e-4_random_uniform_seed44 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --gs_grad_accum_steps 8 \
    --seed 44

# bbh prompt24 lr3e-4 random uniform seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_prompt_time/test_time_evaluate.py \
    --tag bbh_prompt24_lr3e-4_random_uniform_seed44 \
    --gs_epochs 24 \
    --gs_lr 3e-4 \
    --gs_dropout none \
    --gs_batch_size 2 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --gs_grad_accum_steps 8 \
    --seed 44

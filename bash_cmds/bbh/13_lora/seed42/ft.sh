MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# ft lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 8 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 1e-5 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 12 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 1e-5 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 16 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 1e-5 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 1e-5 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42




# ft lr5e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 8 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 5e-5 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 12 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 5e-5 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 16 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 5e-5 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 5e-5 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42







# ft lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 8 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 1e-4 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 12 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 1e-4 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 16 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 1e-4 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --zero_shot \
    --gs_epochs 20 \
    --gs_batch_size 2 \
    --gs_dropout none \
    --gs_ft \
    --gs_ft_lr 1e-4 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --gs_freeze_instruct \
    --seed 42

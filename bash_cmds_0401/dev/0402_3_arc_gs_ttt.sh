# gs
accelerate launch --main_process_port $MASTER_PORT inference_arc/test_time_evaluate.py \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_bos \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --gs_batch_size 1000 \
    --gs_iters 10 \
    --gs_lr 1e-2

# gs with lora
accelerate launch --main_process_port $MASTER_PORT inference_arc/test_time_evaluate.py \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_bos \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --gs_batch_size 1000 \
    --gs_iters 10 \
    --gs_lr 1e-2 \
    --gs_lora \
    --gs_lora_lr 1e-3

# ttt
accelerate launch --main_process_port $MASTER_PORT inference_arc/test_time_evaluate.py \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_bos \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --ttt_iters 10 \
    --ttt_lr 1e-3

# ttt then gs
accelerate launch --main_process_port $MASTER_PORT inference_arc/test_time_evaluate.py \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_bos \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --gs_batch_size 1000 \
    --gs_iters 10 \
    --gs_lr 1e-2 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --ttt_iters 10 \
    --ttt_lr 1e-3

# gs then gs memory check on a100
accelerate launch --main_process_port $MASTER_PORT inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --gs_batch_size 1000 \
    --gs_iters 2 \
    --gs_lr 1e-2 \
    --debug_max_len

# ttt then gs memory check on a100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --gs_batch_size 1000 \
    --gs_iters 10 \
    --gs_lr 1e-2 \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --ttt_iters 10 \
    --ttt_lr 1e-3 \
    --debug_max_len

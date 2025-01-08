# torch.Size([2, 569])
# torch.Size([2, 143])

# debug overfit 1 token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1

# debug overfit 2 token (did not work)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test2 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 1000 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 2

# debug overfit 4 token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test4 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 1000 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 4


# debug multiepoch and stuff
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 2 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1 \
    --wandb

# idk
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 2 \
    --samples_per_epoch 160 \
    --wandb

# 5 and 3 eval samples
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_sub/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_sub/evaluation \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 16 \
    --lr_embedding 0.0 \
    --lr_other 0.0

# debug overfit 1 token compact grids
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1 \
    --compact_grids

# memory stress test (62.3GB)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 4 \
    --num_virtual_tokens 1 \
    --dummy_seq_enc_len 8192 \
    --dummy_seq_dec_len 3613 \
    --flash_attn


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --encoder_name meta-llama/Llama-3.2-3B-Instruct \
    --decoder_name meta-llama/Llama-3.2-3B-Instruct \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 5000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --num_virtual_tokens 6 \
    --dummy_seq_enc_len 569 \
    --dummy_seq_dec_len 143


# memory stress test llama3 (62.3GB)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --encoder_name meta-llama/Llama-3.2-3B-Instruct \
    --decoder_name meta-llama/Llama-3.2-3B-Instruct \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/evaluation \
    --eval_epochs 1 \
    --num_epochs 10 \
    --samples_per_epoch 5000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --dummy_seq_enc_len 569 \
    --dummy_seq_dec_len 143

# debug overfit 1 token llama3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --encoder_name meta-llama/Llama-3.2-3B-Instruct \
    --decoder_name meta-llama/Llama-3.2-3B-Instruct \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1

# debug overfit 1 token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1

# debug overfit 1 token lmheads flashattn
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1 \
    --encoder_lm_head \
    --decoder_lm_head \
    --flash_attn
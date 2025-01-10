# torch.Size([2, 569])
# torch.Size([2, 143])

# debug overfit 1 token (require SGD)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 2 \
    --max_grad_norm 1e8 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --num_workers 0 \
    --encoder_pad_side left \
    --decoder_pad_side right \
    --decoder_gen_pad_side left \
    --debug_random_pad \
    --no_project_kv

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 2 \
    --max_grad_norm 1e8 \
    --no_project_kv \
    --compact_grids
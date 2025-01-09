# torch.Size([2, 569])
# torch.Size([2, 143])

# debug overfit 1 token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit2/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit2/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit2/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --lr_embedding 1e-4 \
    --lr_other 1e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1 \
    --max_grad_norm 1e8 \
    --wandb \
    --num_epochs 1 \
    --samples_per_epoch 500

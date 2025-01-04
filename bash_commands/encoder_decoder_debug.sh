accelerate launch --mixed_precision bf16 encoder_decoder/train.py --tag test --debug

# debug
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --num_epochs 1 \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0 \
    --samples_per_epoch 500

# debug overfit
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 5 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-5 \
    --lr_other 1e-4 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1
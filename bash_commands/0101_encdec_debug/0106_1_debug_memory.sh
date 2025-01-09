# torch.Size([2, 569])
# torch.Size([2, 143])

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --encoder_name llama8b \
    --encoder_name llama8b \
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
    --num_virtual_tokens 1 \
    --trainable_nbit 16 \
    --untrainable_nbit 3.6

# debug overfit 1 token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --encoder_name llama8b \
    --decoder_name llama8b \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 10000 \
    --grad_accum_steps 4 \
    --num_virtual_tokens 1 \
    --dummy_seq_enc_len 128 \
    --dummy_seq_dec_len 128 \
    --trainable_nbit 16 \
    --untrainable_nbit 3.6
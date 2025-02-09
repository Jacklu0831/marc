# try at fitting llama8b with hidden2prompt
# notiemodels, 4bit, 512 tokens, hidden2prompt shared vae -> 64GB
# note full proj is tooooooo many params with 512 tokens
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --samples_per_epoch 64 \
    --conditioning_method hidden2prompt \
    --projection_type shared \
    --vae \
    --ntokens 512 \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --debug_enc_len 5120 \
    --encoder_name llama8b \
    --decoder_name llama8b \
    --untrainable_nbit 4 \
    --trainable_nbit 16 \
    --log_every 1

# every trick at fitting llama8b with prefix2prefix (no projection), it doesnt fit lmao
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --samples_per_epoch 64 \
    --conditioning_method prefix2prefix \
    --ntokens 1 \
    --flash_attn \
    --tie_models \
    --compact_grids \
    --max_seq_len 4096 \
    --debug_enc_len 4096 \
    --encoder_name llama8b \
    --decoder_name llama8b \
    --untrainable_nbit 3.6 \
    --log_every 1
# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0310_noprogram/0310_1_precise.sh

# noprogram precise nobos
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0310_noprogram_precise_nobos \
    --no_bos \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --wandb

# Submitted batch job 58160592
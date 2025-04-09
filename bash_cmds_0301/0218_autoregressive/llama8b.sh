# python make_sbatch.py --ngpu 4 --time 96 --bash_files bash_commands/0218_autoregressive/0218_10_llama8b.sh

# ar llama8b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_8b \
    --model_name llama8b \
    --untrainable_nbit 4 \
    --gradient_checkpointing \
    --train_batch_size 2 \
    --grad_accum_steps 4 \
    --wandb

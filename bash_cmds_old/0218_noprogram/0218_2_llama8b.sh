# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0218_noprogram/0218_2_llama8b.sh --burst

# noprogram llama8b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0218_noprogram_8b \
    --model_name llama8b \
    --untrainable_nbit 4 \
    --gradient_checkpointing \
    --train_batch_size 2 \
    --grad_accum_steps 4 \
    --eval_batch_size 8 \
    --wandb

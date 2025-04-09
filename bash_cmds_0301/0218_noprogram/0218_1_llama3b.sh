# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0218_noprogram/0218_1_llama3b.sh --burst

# noprogram llama3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0218_noprogram_3b \
    --model_name llama3b \
    --untrainable_nbit 4 \
    --gradient_checkpointing \
    --train_batch_size 4 \
    --grad_accum_steps 2 \
    --eval_batch_size 16 \
    --wandb

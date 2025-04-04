# python make_sbatch.py --ngpu 2 --time 24 --gb 64 --bash_files bash_commands/0306_noprogram_nlp/0311_1_gpt2.sh --burst

# noprogram nlp gpt2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0311_noprogram_nlp_gpt2 \
    --eval_epochs 1 \
    --model_name gpt2 \
    --no_flash_attn \
    --wandb

# Submitted batch job 36846
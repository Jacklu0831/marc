# new embedding
accelerate launch --main_process_port 64430 --mixed_precision bf16 encoder_decoder_singleprogram_custom_embedding/train.py \
    --tag new_embedding --wandb\


# new embedding baseline single program 
accelerate launch --main_process_port 55126 --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag single_program_wo_new_embed --wandb\


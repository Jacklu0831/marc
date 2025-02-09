# zb_try
accelerate launch --main_process_port 44046 --mixed_precision bf16 encoder_decoder_singleprogram_custom_embedding/train.py --tag test --wandb


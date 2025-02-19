# new embedding
accelerate launch --main_process_port 27299 --mixed_precision bf16 /scratch/zy3101/marc/encoder_decoder_noprogram/train.py --tag test --thinking_tokens --ntokens 64 --debug
# new embedding baseline single program 
accelerate launch --main_process_port 60817 --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --ntokens 64\
    --tag noprogram_test_breakpoint_recovery --wandb\
    --resume\
    --debug\
    --resume_debug\

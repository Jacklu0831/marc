# new embedding baseline single program 
accelerate launch --main_process_port 55126 --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --ntokens 64\
    --tag noprogram_test_breakpoint_recovery --wandb\
    # --resume\
# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_8_limitinference.sh

# ar extrainference2 limitinference
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0219/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extrainference2 \
    --extra_inference_pairs 2 \
    --limit_inference_pairs \
    --wandb

# ar extrainference100 limitinference
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0219/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extrainference100 \
    --extra_inference_pairs 100 \
    --limit_inference_pairs \
    --wandb

# Submitted batch job 57429194
# Submitted batch job 57429195
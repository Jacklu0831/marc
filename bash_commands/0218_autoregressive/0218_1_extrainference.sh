# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_1_extrainference.sh

# ar extrainference2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extrainference2 \
    --extra_inference_pairs 2 \
    --wandb

# ar extrainference5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extrainference5 \
    --extra_inference_pairs 5 \
    --wandb

# ar extrainference10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extrainference10 \
    --extra_inference_pairs 10 \
    --wandb

# Submitted batch job 57364999
# Submitted batch job 57365000
# Submitted batch job 57365001
# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_17_extrainference.sh

# ar extrainference2 limitinference
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0218/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extrainference2_limitinference \
    --extra_inference_pairs 2 \
    --limit_inference_pairs \
    --wandb

# ar extrainference5 limitinference
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0218/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extrainference5_limitinference \
    --extra_inference_pairs 5 \
    --limit_inference_pairs \
    --wandb

# ar extrainference10 limitinference
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0218/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extrainference10_limitinference \
    --limit_inference_pairs \
    --extra_inference_pairs 10 \
    --wandb

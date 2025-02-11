# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0209_autoregressive/0209_7_reducelr.sh

# ar lr7.5e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_lr7.5e-5 \
    --train_no_sample \
    --eval_no_sample \
    --lr_embedding 7.5e-5 \
    --lr_program 7.5e-5 \
    --lr_prior 7.5e-5 \
    --lr_other 7.5e-5 \
    --wandb

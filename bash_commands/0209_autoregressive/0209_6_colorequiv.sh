# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0209_autoregressive/0209_6_colorequiv.sh

# ar colorequiv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_colorequiv \
    --train_no_sample \
    --eval_no_sample \
    --color_equiv \
    --wandb

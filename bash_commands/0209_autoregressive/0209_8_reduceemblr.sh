# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0209_autoregressive/0209_8_reduceemblr.sh

# ar emblr1e-5 programlr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_emblr1e-5_programlr1e-5 \
    --train_no_sample \
    --eval_no_sample \
    --lr_embedding 1e-5 \
    --lr_program 1e-5 \
    --lr_prior 1e-5 \
    --wandb

# ar emblr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_emblr1e-5 \
    --train_no_sample \
    --eval_no_sample \
    --lr_embedding 1e-5 \
    --wandb

# ar emblr1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_emblr1e-6 \
    --train_no_sample \
    --eval_no_sample \
    --lr_embedding 1e-6 \
    --wandb

# ar emblr0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_emblr0 \
    --train_no_sample \
    --eval_no_sample \
    --lr_embedding 0 \
    --wandb

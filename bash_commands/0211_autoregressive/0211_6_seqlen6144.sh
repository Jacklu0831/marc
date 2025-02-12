# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_6_seqlen6144.sh

# ar seqlen6144
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_seqlen6144 \
    --max_seq_len 6144 \
    --wandb

# Submitted batch job 57179507
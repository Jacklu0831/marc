# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_8_evalbatchsize1.sh

# ar evalbatchsize1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_evalbatchsize1 \
    --eval_batch_size 1 \
    --wandb

# Submitted batch job 57180930
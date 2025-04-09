# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_12_nooverfit.sh

# ar seqlen8192
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0213/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_seqlen8192 \
    --max_seq_len 8192 \
    --wandb

# ar weightdecay0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0213/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_weightdecay0.01 \
    --weight_decay 0.01 \
    --wandb

# ar dropout0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0213/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_dropout0.1 \
    --lora_dropout 0.1 \
    --wandb

# Submitted batch job 57244789
# Submitted batch job 57244790
# Submitted batch job 57244791
# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0317_noprogram/0317_1_ntoken.sh --burst

# noprogram ntoken16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_ntoken16 \
    --ntoken 16 \
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --wandb

# noprogram ntoken16 attentioncutoff
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_ntoken16_attentioncutoff \
    --ntoken 16 \
    --attention_cutoff \
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --wandb

# noprogram ntoken16 attentioncutoff attendprevprograms
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_ntoken16_attentioncutoff_attendprevprograms \
    --ntoken 16 \
    --attention_cutoff \
    --attend_prev_programs \
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --wandb

# Submitted batch job 38666
# Submitted batch job 38667
# Submitted batch job 38668
# python make_sbatch.py --ngpu 2 --time 24 --bash_files bash_commands/0306_noprogram_nlp/0310_0_base.sh --burst

# noprogram nlp base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0310_noprogram_nlp_base \
    --wandb

# noprogram nlp excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0310_noprogram_nlp_excludefirst \
    --loss_type exclude_first \
    --wandb

# noprogram nlp delimiteranswer
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0310_noprogram_nlp_delimiteranswer \
    --delimiter " Answer: " \
    --wandb

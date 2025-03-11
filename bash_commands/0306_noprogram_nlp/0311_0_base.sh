# python make_sbatch.py --ngpu 2 --time 24 --gb 64 --bash_files bash_commands/0306_noprogram_nlp/0311_0_base.sh --burst

# noprogram nlp base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0311_noprogram_nlp_base \
    --eval_epochs 1 \
    --wandb

# noprogram nlp excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0311_noprogram_nlp_excludefirst \
    --loss_type exclude_first \
    --eval_epochs 1 \
    --wandb

# noprogram nlp delimiteranswer
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0311_noprogram_nlp_delimiteranswer \
    --delimiter ' Answer: ' \
    --eval_epochs 1 \
    --wandb

# failed
# Submitted batch job 36682
# Submitted batch job 36683
# Submitted batch job 36684

# same thing just eval per epoch
# Submitted batch job 36765
# Submitted batch job 36766
# Submitted batch job 36767
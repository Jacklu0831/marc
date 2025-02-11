# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_0_base.sh

# ar base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_base \
    --wandb

# ar seqlen7168
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_seqlen7168 \
    --max_seq_len 7168 \
    --wandb

# ar ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_ntokens128 \
    --ntokens 128 \
    --wandb

# ar colorequiv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_colorequiv \
    --wandb

# ar noaug
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_noaug \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --wandb

# ar gradaccum8 lr2e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_gradaccum8_lr2e-4 \
    --grad_accum_steps 8 \
    --lr_embedding 2e-5 \
    --lr_program 2e-4 \
    --lr_prior 2e-4 \
    --lr_other 2e-4 \
    --wandb

# ar nodim
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_nodim \
    --no_dim \
    --wandb

# ar norm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_norm \
    --normalize \
    --wandb

# ar separate color tokens
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_separate_color_tokens \
    --separate_color_tokens \
    --wandb

# Submitted batch job 57149131
# Submitted batch job 57149132
# Submitted batch job 57149133
# Submitted batch job 57149134
# Submitted batch job 57149135
# Submitted batch job 57149136
# Submitted batch job 57149137
# Submitted batch job 57149138
# Submitted batch job 57149139
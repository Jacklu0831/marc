# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0209_autoregressive/0209_0_base.sh

# ar base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_base \
    --train_no_sample \
    --eval_no_sample \
    --wandb

# ar noflashattn
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_noflashattn \
    --train_no_sample \
    --eval_no_sample \
    --no_flash_attn \
    --wandb

# ar colorequiv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_colorequiv \
    --train_no_sample \
    --eval_no_sample \
    --color_equiv \
    --wandb

# ar norm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_norm \
    --train_no_sample \
    --eval_no_sample \
    --normalize \
    --wandb

# ar residual
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_residual \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --wandb

# ar residual norm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_residual_norm \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --normalize \
    --wandb

# ar reduceemblr
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_reduceemblr \
    --train_no_sample \
    --eval_no_sample \
    --lr_embedding 1e-5 \
    --wandb

# ar reduceemblr_reduceprogramlr
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_reduceemblr_reduceprogramlr \
    --train_no_sample \
    --eval_no_sample \
    --lr_embedding 1e-5 \
    --lr_program 1e-5 \
    --lr_prior 1e-5 \
    --wandb

# ar curriculum2000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_curriculum2000 \
    --train_no_sample \
    --eval_no_sample \
    --curriculum_iters 2000 \
    --min_num_pair 3 \
    --wandb

# ar curriculum2000 nowarmup
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_curriculum2000_nowarmup \
    --train_no_sample \
    --eval_no_sample \
    --curriculum_iters 2000 \
    --min_num_pair 3 \
    --warmup_epoch 0 \
    --wandb

# ar noaug
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_noaug \
    --train_no_sample \
    --eval_no_sample \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --wandb

# ar nod8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_nod8 \
    --train_no_sample \
    --eval_no_sample \
    --no_color_permute \
    --no_pair_permute \
    --wandb

# Submitted batch job 57077640 # base loss jump
# Submitted batch job 57077641
# Submitted batch job 57077642
# Submitted batch job 57077643 # cancelled to use float32 norm
# Submitted batch job 57077644
# Submitted batch job 57088140 # cancelled to use float32 norm
# Submitted batch job 57077645
# Submitted batch job 57077646 # reduceemblr_reduceprogramlr loss jump
# Submitted batch job 57077647 # currulum sucked
# Submitted batch job 57077648 # currulum sucked
# Submitted batch job 57077649
# Submitted batch job 57077650
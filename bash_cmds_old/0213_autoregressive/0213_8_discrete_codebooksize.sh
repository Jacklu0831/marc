# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_8_discrete_codebooksize.sh

# ar codesize1024 wait2epochs_then_warmup10epochs ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0216/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_codesize1024_wait2epochs_then_warmup10epochs_ntokens128 \
    --ntokens 128 \
    --normalize \
    --codebook_size 1024 \
    --commitment_loss_offset_epochs 2 \
    --commitment_loss_linear_epochs 10 \
    --codebook_loss_offset_epochs 2 \
    --codebook_loss_linear_epochs 10 \
    --wandb

# ar codesize1024 warmup10epochs ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0216/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_codesize1024_warmuplambda_ntokens128 \
    --ntokens 128 \
    --normalize \
    --codebook_size 1024 \
    --commitment_loss_linear_epochs 10 \
    --codebook_loss_linear_epochs 10 \
    --wandb




# ar codesize256 codebook2epochs_then_warmup10epochs ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0216/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_codesize256_codebook2epochs_then_warmup10epochs_ntokens128 \
    --ntokens 128 \
    --normalize \
    --codebook_size 256 \
    --commitment_loss_offset_epochs 2 \
    --commitment_loss_linear_epochs 10 \
    --warmup_cookbook_only_epochs 2 \
    --wandb

# ar codesize1024 codebook2epochs_then_warmup10epochs ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0216/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_codesize1024_codebook2epochs_then_warmup10epochs_ntokens128 \
    --ntokens 128 \
    --normalize \
    --codebook_size 1024 \
    --commitment_loss_offset_epochs 2 \
    --commitment_loss_linear_epochs 10 \
    --warmup_cookbook_only_epochs 2 \
    --wandb

# ar codesize2048 codebook2epochs_then_warmup10epochs ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0216/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_codesize2048_codebook2epochs_then_warmup10epochs_ntokens128 \
    --ntokens 128 \
    --normalize \
    --codebook_size 2048 \
    --commitment_loss_offset_epochs 2 \
    --commitment_loss_linear_epochs 10 \
    --warmup_cookbook_only_epochs 2 \
    --wandb

# ar codesize4096 codebook2epochs_then_warmup10epochs ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0216/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_codesize4096_codebook2epochs_then_warmup10epochs_ntokens128 \
    --ntokens 128 \
    --normalize \
    --codebook_size 4096 \
    --commitment_loss_offset_epochs 2 \
    --commitment_loss_linear_epochs 10 \
    --warmup_cookbook_only_epochs 2 \
    --wandb




# ar codesize1024 codebook2epochs_then_warmup10epochs_noprior ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0216/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_codesize1024_codebook2epochs_then_warmup10epochs_noprior_ntokens128 \
    --ntokens 128 \
    --normalize \
    --codebook_size 1024 \
    --commitment_loss_offset_epochs 2 \
    --commitment_loss_linear_epochs 10 \
    --warmup_cookbook_only_epochs 2 \
    --no_discrete_prior \
    --wandb

# Submitted batch job 57328335
# Submitted batch job 57328336

# Submitted batch job 57328337
# Submitted batch job 57328334
# Submitted batch job 57328338
# Submitted batch job 57328339

# Submitted batch job 57328340
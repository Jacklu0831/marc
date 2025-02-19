# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_4_discrete_commitment_loss.sh

# ar codesize1024 commit0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0211/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_codesize1024_commit0.05 \
    --codebook_size 1024 \
    --commitment_loss_lambda 0.05 \
    --wandb

# ar codesize1024 commit0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0211/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_codesize1024_commit0.1 \
    --codebook_size 1024 \
    --commitment_loss_lambda 0.1 \
    --wandb

# ar codesize1024 commit0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0211/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_codesize1024_commit0.5 \
    --codebook_size 1024 \
    --commitment_loss_lambda 0.5 \
    --wandb

# Submitted batch job 57169173
# Submitted batch job 57169174
# Submitted batch job 57169175
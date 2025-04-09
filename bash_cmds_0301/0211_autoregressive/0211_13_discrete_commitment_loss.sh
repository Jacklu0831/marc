# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_13_discrete_commitment_loss.sh

# ar projnone codesize4096 commit0.001
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_projnone_codesize4096_commit0.001 \
    --projection_type none \
    --codebook_size 4096 \
    --commitment_loss_lambda 0.001 \
    --wandb

# ar projnone codesize1024 commit0.001
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_projnone_codesize1024_commit0.001 \
    --projection_type none \
    --codebook_size 1024 \
    --commitment_loss_lambda 0.001 \
    --wandb

# ar projnone codesize1024 commit0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_projnone_codesize1024_commit0.01 \
    --projection_type none \
    --codebook_size 1024 \
    --commitment_loss_lambda 0.01 \
    --wandb

# ar projnone codesize1024 commit0.01 norm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_projnone_codesize1024_commit0.01_norm \
    --projection_type none \
    --codebook_size 1024 \
    --commitment_loss_lambda 0.01 \
    --normalize \
    --wandb

# ar projnone codesize1024 commit0.01 linear
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_projnone_codesize1024_commit0.01_linear \
    --projection_type none \
    --codebook_size 1024 \
    --commitment_loss_lambda 0.01 \
    --linear_commitment \
    --wandb

# Submitted batch job 57244805
# Submitted batch job 57244806
# Submitted batch job 57244807
# Submitted batch job 57244808
# Submitted batch job 57244809
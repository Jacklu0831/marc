# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_3_discrete_codebooksize.sh

# ar codesize1024 commit0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_codesize1024_commit0.25 \
    --codebook_size 1024 \
    --commitment_loss_lambda 0.25 \
    --wandb

# ar codesize512 commit0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_codesize512_commit0.25 \
    --codebook_size 512 \
    --commitment_loss_lambda 0.25 \
    --wandb

# ar codesize256 commit0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_codesize256_commit0.25 \
    --codebook_size 256 \
    --commitment_loss_lambda 0.25 \
    --wandb

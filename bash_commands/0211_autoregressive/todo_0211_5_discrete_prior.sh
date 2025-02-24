# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_5_discrete_prior.sh

# ar codesize1024 commit0.25 discreteprior
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0211/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_codesize1024_commit0.25_discreteprior \
    --codebook_size 1024 \
    --commitment_loss_lambda 0.25 \
    --discrete_prior \
    --wandb

# Submitted batch job 57230223
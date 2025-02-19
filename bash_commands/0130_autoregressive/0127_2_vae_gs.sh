# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0127_2_vae_gs.sh

# ar kl1e-5 gs0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0204/train.py \
    --tag 0202_ar_kl1e-4_gs0 \
    --kl_loss_lambda 1e-5 \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 0 \
    --wandb

# ar kl1e-5 gs10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0204/train.py \
    --tag 0202_ar_kl1e-4_gs10 \
    --kl_loss_lambda 1e-5 \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 10 \
    --wandb

# ar kl1e-5 gs50
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0204/train.py \
    --tag 0202_ar_kl1e-4_gs50 \
    --kl_loss_lambda 1e-5 \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 50 \
    --wandb

# ar kl1e-5 gs100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0204/train.py \
    --tag 0202_ar_kl1e-4_gs100 \
    --kl_loss_lambda 1e-5 \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 100 \
    --wandb

# Submitted batch job 57005226
# Submitted batch job 57005227
# Submitted batch job 57005228
# Submitted batch job 57005229
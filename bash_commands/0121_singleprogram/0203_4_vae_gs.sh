# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_4_vae_gs.sh

# single vae full lambda1e-4 gs0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0205/train.py \
    --tag 0203_single_vae_full_lambda1e-4_gs0 \
    --vae \
    --projection_type full \
    --kl_loss_lambda 1e-4 \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 0 \
    --wandb

# single vae full lambda1e-4 gs10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0205/train.py \
    --tag 0203_single_vae_full_lambda1e-4_gs10 \
    --vae \
    --projection_type full \
    --kl_loss_lambda 1e-4 \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 10 \
    --wandb

# single vae full lambda1e-4 gs50
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0205/train.py \
    --tag 0203_single_vae_full_lambda1e-4_gs50 \
    --vae \
    --projection_type full \
    --kl_loss_lambda 1e-4 \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 50 \
    --wandb

# single vae full lambda1e-4 gs100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0205/train.py \
    --tag 0203_single_vae_full_lambda1e-4_gs100 \
    --vae \
    --projection_type full \
    --kl_loss_lambda 1e-4 \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 100 \
    --wandb

# Submitted batch job 56990576
# Submitted batch job 56990577
# Submitted batch job 56990578
# Submitted batch job 56990579
# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0127_1_other.sh

# ar novae
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_novae \
    --train_no_sample \
    --eval_no_sample \
    --wandb

# ar nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_nolora \
    --no_lora \
    --wandb

# ar novae_voting
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --tag 0202_ar_novae_voting \
    --no_lora \
    --wandb

# ar kl1e-5_evalnosample
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_kl1e-5_evalnosample \
    --kl_loss_lambda 1e-5 \
    --eval_no_sample \
    --wandb

# Submitted batch job 56865479
# Submitted batch job 56865480
# Submitted batch job 56865481
# Submitted batch job 56865482
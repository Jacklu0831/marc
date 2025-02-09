# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0206_0_highlr.sh
# run this after knowing how many warmup epoch

# ar novae lr1.5e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0204/train.py \
    --tag 0206_ar_novae_lr1.5e-4 \
    --train_no_sample \
    --eval_no_sample \
    --lr_other 1.5e-4 \
    --lr_embedding 1.5e-5 \
    --wandb

# ar novae lr2e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0204/train.py \
    --tag 0206_ar_novae_lr2e-4 \
    --train_no_sample \
    --eval_no_sample \
    --lr_other 2e-4 \
    --lr_embedding 2e-5 \
    --lr_program 2e-5 \
    --wandb

# Submitted batch job 57068591
# Submitted batch job 57068592
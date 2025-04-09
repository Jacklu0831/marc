# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0206_1_moreepoch.sh

# ar novae epoch30
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0204/train.py \
    --tag 0206_ar_novae_epoch30 \
    --train_no_sample \
    --eval_no_sample \
    --num_epochs 30 \
    --wandb

# ar novae epoch35
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0204/train.py \
    --tag 0206_ar_novae_epoch35 \
    --train_no_sample \
    --eval_no_sample \
    --num_epochs 35 \
    --wandb

# Submitted batch job 57068582
# Submitted batch job 57068583
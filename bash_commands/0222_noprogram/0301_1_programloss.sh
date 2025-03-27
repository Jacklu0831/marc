# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0222_noprogram/0301_1_programloss.sh

# noprogram programloss0.3concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntokens16_programloss0.3concat \
    --program_type concat \
    --program_loss_lambda 0.3 \
    --min_num_pair 8 \
    --ntokens 16 \
    --wandb

# noprogram programloss1.0concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntokens16_programloss1.0concat \
    --program_type concat \
    --program_loss_lambda 1.0 \
    --min_num_pair 8 \
    --ntokens 16 \
    --wandb

# noprogram programloss0.3random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntokens16_programloss0.3random \
    --program_type random \
    --program_loss_lambda 0.3 \
    --min_num_pair 8 \
    --ntokens 16 \
    --wandb

# noprogram programloss1.0random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntokens16_programloss1.0random \
    --program_type random \
    --program_loss_lambda 1.0 \
    --min_num_pair 8 \
    --ntokens 16 \
    --wandb

# Submitted batch job 57867440 # mysterious assertion failure
# Submitted batch job 57867441
# Submitted batch job 57867442
# Submitted batch job 57867443
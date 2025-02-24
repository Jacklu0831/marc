# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_6_curriculum.sh
# 20000 samples per batch / global batch size 16 = 1250
# we have pairs 3 4 5 6 7 8 -> 1250 / 6 ~= 200 curriculum iter max to finish curriculum learning in one epoch

# ar curriculum200
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_curriculum200 \
    --curriculum_iters 200 \
    --min_num_pair 3 \
    --warmup_epoch 1 \
    --wandb

# ar curriculum200 nowarmup
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_curriculum200_nowarmup \
    --curriculum_iters 200 \
    --min_num_pair 3 \
    --warmup_epoch 0 \
    --wandb

# Submitted batch job 57246080
# Submitted batch job 57246081
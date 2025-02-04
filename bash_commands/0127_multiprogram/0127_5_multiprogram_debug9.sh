# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0127_multiprogram/0127_5_multiprogram_debug9.sh

# nobasicaug_notrainoriginal_evalbs1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_multiprogram/train.py \
    --tag 0127_multi_nobasicaug_notrainoriginal_evalbs1 \
    --no_color_permute \
    --no_d8 \
    --no_pair_permute \
    --no_train_original \
    --eval_batch_size 1 \
    --wandb

# nobasicaug_notrainoriginal_evalbs2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_multiprogram/train.py \
    --tag 0127_multi_nobasicaug_notrainoriginal_evalbs2 \
    --no_color_permute \
    --no_d8 \
    --no_pair_permute \
    --no_train_original \
    --eval_batch_size 2 \
    --wandb

# nobasicaug_notrainoriginal_evalbs4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_multiprogram/train.py \
    --tag 0127_multi_nobasicaug_notrainoriginal_evalbs4 \
    --no_color_permute \
    --no_d8 \
    --no_pair_permute \
    --no_train_original \
    --eval_batch_size 4 \
    --wandb

# early stop, all got slightly lower than base in debug10 but about same
# Submitted batch job 56820645
# Submitted batch job 56820646
# Submitted batch job 56820647
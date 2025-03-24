# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_noprogram/0317_0_base.sh
# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0317_noprogram/0317_0_base.sh

# noprogram base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_base \
    --wandb

# noprogram repro
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_repro \
    --wandb

# noprogram nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_nolora \
    --no_lora \
    --wandb

# noprogram minnumpair3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --min_num_pair 3 \
    --tag 0317_noprogram_minnumpair3 \
    --wandb

# noprogram 100task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_100task \
    --train_data_dir ./data/re-arc/train_data_100/tasks \
    --wandb

# noprogram 200task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_200task \
    --train_data_dir ./data/re-arc/train_data_200/tasks \
    --wandb

# noprogram noaug
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_noaug \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --wandb

# noprogram extraaug0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_extraaug0.3 \
    --extra_augment_ratio 0.3 \
    --wandb

# noprogram extraaug0.3single
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_extraaug0.3single \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --wandb

# noprogram nomaxseqlen
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_nomaxseqlen \
    --max_seq_len 10000000 \
    --wandb

# noprogram llama3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_llama3b \
    --model_name llama3b \
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --untrainable_nbit 4 \
    --wandb


# Submitted batch job 58466746
# Submitted batch job 58466747
# Submitted batch job 58466748
# Submitted batch job 58466749
# Submitted batch job 58466750
# Submitted batch job 58466751
# Submitted batch job 58466752
# Submitted batch job 58466753
# Submitted batch job 58466754
# Submitted batch job 58466755
# Submitted batch job 58466756

# resume llama3b, 100task
# Submitted batch job 58648636
# Submitted batch job 58698155
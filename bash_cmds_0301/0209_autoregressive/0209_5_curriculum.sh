# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0209_autoregressive/0209_5_curriculum.sh
# 20000 samples per batch / global batch size 16 = 1250
# we have pairs 3 4 5 6 7 8 -> 1250 / 6 ~= 200 curriculum iter max to finish curriculum learning in one epoch

# ar curriculum100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_curriculum100 \
    --train_no_sample \
    --eval_no_sample \
    --curriculum_iters 100 \
    --min_num_pair 3 \
    --wandb

# ar curriculum200
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_curriculum200 \
    --train_no_sample \
    --eval_no_sample \
    --curriculum_iters 200 \
    --min_num_pair 3 \
    --wandb

# ar curriculum100 nowarmup
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_curriculum100_nowarmup \
    --train_no_sample \
    --eval_no_sample \
    --curriculum_iters 100 \
    --min_num_pair 3 \
    --warmup_epoch 0 \
    --wandb

# ar curriculum200 nowarmup
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag 0211_ar_curriculum200_nowarmup \
    --train_no_sample \
    --eval_no_sample \
    --curriculum_iters 200 \
    --min_num_pair 3 \
    --warmup_epoch 0 \
    --wandb

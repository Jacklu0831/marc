# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0207_1_gs.sh

# single gs1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0207/train.py \
    --tag 0207_single_gs1 \
    --eval_batch_size 1 \
    --eval_gs_batch_size 10000 \
    --eval_gs_take_best \
    --eval_gs_iters 1 \
    --wandb

# single gs5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0207/train.py \
    --tag 0207_single_gs5 \
    --eval_batch_size 1 \
    --eval_gs_batch_size 10000 \
    --eval_gs_take_best \
    --eval_gs_iters 5 \
    --wandb

# single gs25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0207/train.py \
    --tag 0207_single_gs25 \
    --eval_batch_size 1 \
    --eval_gs_batch_size 10000 \
    --eval_gs_take_best \
    --eval_gs_iters 25 \
    --wandb

# single gs100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0207/train.py \
    --tag 0207_single_gs100 \
    --eval_batch_size 1 \
    --eval_gs_batch_size 10000 \
    --eval_gs_take_best \
    --eval_gs_iters 500 \
    --wandb

# single gs200
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0207/train.py \
    --tag 0207_single_gs200 \
    --eval_batch_size 1 \
    --eval_gs_batch_size 10000 \
    --eval_gs_take_best \
    --eval_gs_iters 200 \
    --wandb

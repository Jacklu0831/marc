# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0110_multigpu/0112_4_llama1b3b.sh
# using flashattn because fast
# when training 1bto3b or 3bto1b, the projection matrix is necessary (only 0.14GB for the default ntoken8)

# 1bto3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0112_1bto3b \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full \
    --encoder_gradient_checkpointing \
    --encoder_name llama1b \
    --decoder_name llama3b \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --flash_attn \
    --wandb

# 3bto1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0112_3bto1b \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full \
    --encoder_gradient_checkpointing \
    --encoder_name llama3b \
    --decoder_name llama1b \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --flash_attn \
    --wandb

# kept these, but might run into gpu synchronization error
# Submitted batch job 55777131
# Submitted batch job 55777132
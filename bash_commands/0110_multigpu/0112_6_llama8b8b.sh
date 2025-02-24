# python make_sbatch.py --ngpu 4 --time 96 --bash_files bash_commands/0110_multigpu/0112_6_llama8b8b.sh
# using flashattn because fast
# when training 8bto1b or 8bto3b, the projection matrix is necessary (only <0.5GB for the default ntoken8)
# 8bto1b doesnt need decoder ckpting, 58GB and 280hr on single gpu
# 8bto3b needs both ckpting,          51GB and 330hr on single gpu
# 8bto8b needs both ckpting,          57GB and 394hr on single gpu

# 8bto1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0112_8bto1b \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --train_data_dir ./data/re-arc/train_data/tasks \
    --eval_train_dir ./data/re-arc/arc_original/training \
    --eval_eval_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full \
    --encoder_gradient_checkpointing \
    --encoder_name llama8b \
    --decoder_name llama1b \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --flash_attn \
    --wandb

# 8bto3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0112_8bto3b \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --train_data_dir ./data/re-arc/train_data/tasks \
    --eval_train_dir ./data/re-arc/arc_original/training \
    --eval_eval_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --encoder_name llama8b \
    --decoder_name llama3b \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --flash_attn \
    --wandb

# 8bto8b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0112_8bto8b \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --train_data_dir ./data/re-arc/train_data/tasks \
    --eval_train_dir ./data/re-arc/arc_original/training \
    --eval_eval_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --encoder_name llama8b \
    --decoder_name llama8b \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --flash_attn \
    --wandb

# had to stop these due to imminent gpu synchronization error
# Submitted batch job 55777303
# Submitted batch job 55777304
# Submitted batch job 55777305

# Submitted batch job 55792731
# Submitted batch job 55792732
# Submitted batch job 55792726
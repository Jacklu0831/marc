# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0110_multigpu/0110_4_2gpu_quantized.sh

# 2gpu 3.6bit 16bit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0110_2gpu_3.6bit_16bit \
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
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --wandb

# Submitted batch job 55748694
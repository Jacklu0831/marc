# python make_sbatch.py --time 48 --bash_files bash_commands/0101_encdec_debug/0105_debug.sh

# debug invar0.0
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.0 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 10000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --wandb

# debug invar0.001
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.001 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 10000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.001 \
    --wandb

# debug invar0.01
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.01 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 10000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.01 \
    --wandb

# OOM killed
# Submitted batch job 55573972
# Submitted batch job 55573973
# Submitted batch job 55573974

# cancelled
# Submitted batch job 55596264
# Submitted batch job 55596265
# Submitted batch job 55596266
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
    --invar_loss_lambda 0.0 \
    --wandb

# debug invar0.1
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 10000 \
    --invar_loss_lambda 0.1 \
    --wandb

# debug invar0.3
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.3 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 10000 \
    --invar_loss_lambda 0.3 \
    --wandb

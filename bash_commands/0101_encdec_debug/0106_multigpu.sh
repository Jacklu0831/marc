# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0101_encdec_debug/0106_multigpu.sh

# debug invar0.0 multigpu
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.0_multigpu \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 5000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --wandb

# full invar0.0 multigpu
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag full_invar0.0_multigpu \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --wandb

# Submitted batch job 55607768
# Submitted batch job 55607769
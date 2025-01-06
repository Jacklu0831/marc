# python make_sbatch.py --time 48 --bash_files bash_commands/0101_encdec_debug/0105_full.sh

# full invar0.0
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag full_invar0.0 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --invar_loss_lambda 0.0 \
    --wandb

# full invar0.1
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag full_invar0.1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --invar_loss_lambda 0.1 \
    --wandb

# full invar0.3
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag full_invar0.3 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --invar_loss_lambda 0.3 \
    --wandb

# had dataloader bug
# Submitted batch job 55573969
# Submitted batch job 55573970
# Submitted batch job 55573971

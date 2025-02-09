# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0110_multigpu/0111_4_gradnorm.sh

# gradnorm1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0110_gradnorm1 \
    --train_data_dir /scratch/zy3101/re-arc/train_data/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --max_grad_norm 1.0 \
    --wandb

# gradnorm10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0110_gradnorm10 \
    --train_data_dir /scratch/zy3101/re-arc/train_data/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --max_grad_norm 10.0 \
    --wandb

# Submitted batch job 55768965
# Submitted batch job 55768966
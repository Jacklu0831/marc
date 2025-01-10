# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0110_multigpu/0110_2_2gpu_more_invar.sh

# 2gpu invar0.003
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0110_2gpu_invar0.003 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.003 \
    --compact_grids \
    --max_seq_len 5120 \
    --wandb

# 2gpu invar0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0110_2gpu_invar0.01 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.01 \
    --compact_grids \
    --max_seq_len 5120 \
    --wandb

# 2gpu invar0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0110_2gpu_invar0.03 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.03 \
    --compact_grids \
    --max_seq_len 5120 \
    --wandb

# Submitted batch job 55745792
# Submitted batch job 55745793
# Submitted batch job 55745794
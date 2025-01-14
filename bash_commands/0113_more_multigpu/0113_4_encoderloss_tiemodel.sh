# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0113_more_multigpu/0113_4_encoderloss_tiemodel.sh

# encoderloss0.0 tiemodel
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss0.0_tiemodel \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 25 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.03 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method hidden2prompt_full \
    --encoder_loss_lambda 0.0 \
    --tie_models \
    --wandb

# encoderloss1.0 tiemodel
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss1.0_tiemodel \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 25 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.03 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method hidden2prompt_full \
    --encoder_loss_lambda 1.0 \
    --tie_models \
    --wandb

# encoderloss0.0 notiemodel
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss0.0_notiemodel \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 25 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.03 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method hidden2prompt_full \
    --encoder_loss_lambda 0.0 \
    --wandb

# encoderloss1.0 notiemodel
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss1.0_notiemodel \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 25 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.03 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method hidden2prompt_full \
    --encoder_loss_lambda 1.0 \
    --wandb

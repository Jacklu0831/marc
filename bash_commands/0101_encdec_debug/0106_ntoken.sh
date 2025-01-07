# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0101_encdec_debug/0106_ntoken.sh

# debug invar0.0 ntoken1
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.0_ntoken1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 10 \
    --samples_per_epoch 5000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --num_virtual_tokens 1 \
    --wandb

# debug invar0.0 ntoken5
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.0_ntoken5 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 10 \
    --samples_per_epoch 5000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --num_virtual_tokens 5 \
    --wandb

# debug invar0.0 ntoken10
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.0_ntoken10 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 10 \
    --samples_per_epoch 5000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --num_virtual_tokens 10 \
    --wandb

# debug invar0.0 ntoken25
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.0_ntoken25 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 10 \
    --samples_per_epoch 5000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --num_virtual_tokens 25 \
    --wandb

# full invar0.0 ntoken1
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag full_invar0.0_ntoken1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --num_virtual_tokens 1 \
    --wandb

# full invar0.0 ntoken5
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag full_invar0.0_ntoken1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --num_virtual_tokens 5 \
    --wandb

# full invar0.0 ntoken10
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag full_invar0.0_ntoken1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --num_virtual_tokens 10 \
    --wandb

# full invar0.0 ntoken25
accelerate launch --mixed_precision bf16 encoder_decoder/train.py \
    --tag full_invar0.0_ntoken1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --num_virtual_tokens 25 \
    --wandb

# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0101_encdec_debug/0108_0_singlegpu.sh

# singlegpu
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0108_0_singlegpu \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --wandb

# Submitted batch job 55650246
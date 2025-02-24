# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0110_multigpu/0112_0_seqlen4096.sh

# seqlen4096
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0110_seqlen4096 \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --train_data_dir ./data/re-arc/train_data/tasks \
    --eval_train_dir ./data/re-arc/arc_original/training \
    --eval_eval_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 4096 \
    --wandb

# Submitted batch job 55774597
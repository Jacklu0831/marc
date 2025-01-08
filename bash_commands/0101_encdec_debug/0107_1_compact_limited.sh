# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0101_encdec_debug/0107_1_compact_limited.sh

# debug invar0.0 compact limited
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag debug_invar0.0_compact_limited \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug/evaluation \
    --eval_epochs 1 \
    --num_epochs 10 \
    --samples_per_epoch 5000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --no_gradient_checkpointing \
    --wandb

# full invar0.0 compact limited
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag full_invar0.0_compact_limited \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --no_gradient_checkpointing \
    --decoder_lm_head \
    --wandb

# logged to alex
# Submitted batch job 55622318
# Submitted batch job 55622319

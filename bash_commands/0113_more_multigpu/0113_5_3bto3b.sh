# python make_sbatch.py --ngpu 2 --time 96 --bash_files bash_commands/0113_more_multigpu/0113_4_encoderloss_tiemodel.sh

# 3bto3b aug0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_3bto3b_aug0.0 \
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
    --conditioning_method hidden2prompt \
    --encoder_gradient_checkpointing \
    --encoder_name llama3b \
    --decoder_name llama3b \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --tie_models \
    --wandb

# 3bto3b aug0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_3bto3b_aug0.3 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 25 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.3 \
    --invar_loss_lambda 0.03 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method hidden2prompt \
    --encoder_gradient_checkpointing \
    --encoder_name llama3b \
    --decoder_name llama3b \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --tie_models \
    --wandb

# python make_sbatch.py --ngpu 2 --time 96 --bash_files bash_commands/0110_multigpu/0112_5_llama3b3b.sh
# using flashattn because fast
# when training 1bto3b or 3bto1b, the projection matrix is necessary (only 0.14GB for the default ntoken8)

# 3bto3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0112_3bto3b \
    --train_data_dir ./data/re-arc/train_data/tasks \
    --eval_train_dir ./data/re-arc/arc_original/training \
    --eval_eval_dir ./data/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full \
    --encoder_gradient_checkpointing \
    --encoder_name llama3b \
    --decoder_name llama3b \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --flash_attn \
    --wandb

# gpu synchronization error
# Submitted batch job 55777133

# Submitted batch job 55792717
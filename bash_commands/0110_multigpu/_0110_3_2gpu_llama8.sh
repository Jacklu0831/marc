# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0110_multigpu/0110_3_2gpu_llama8.sh

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
    --encoder_name llama3b \
    --decoder_name llama3b \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 1600 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 4096 \
    --dummy_seq_enc_len 4096 \
    --dummy_seq_dec_len 2048 \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --log_every 1 \
    --project_kv \
    --encoder_gradient_checkpointing

# llama8b 8token 512seqlen
# no projectkv 47.4GB, 1024seqlen makes it 70.5GB
# projectkv 64.0GB
# projectkv gradientcheckpointing 54.5GB
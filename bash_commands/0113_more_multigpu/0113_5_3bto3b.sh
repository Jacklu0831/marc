# python make_sbatch.py --ngpu 2 --time 96 --bash_files bash_commands/0113_more_multigpu/0113_4_encoderloss_tiemodel.sh

# 3bto3b aug0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_3bto3b_aug0.0 \
    --augment_ratio 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --encoder_name llama3b \
    --decoder_name llama3b \
    --untrainable_nbit 3.6 \
    --wandb

# 3bto3b aug0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_3bto3b_aug0.3 \
    --augment_ratio 0.3 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --encoder_name llama3b \
    --decoder_name llama3b \
    --untrainable_nbit 3.6 \
    --wandb

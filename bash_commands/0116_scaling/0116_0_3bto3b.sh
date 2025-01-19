# python make_sbatch.py --ngpu 2 --time 96 --bash_files bash_commands/0116_scaling/0116_0_3bto3b.sh

# 3bto3b aug0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new2/train.py \
    --tag 0116_3bto3b_aug0.0 \
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

# 3bto3b notiemodels
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new2/train.py \
    --tag 0116_3bto3b_notiemodels \
    --augment_ratio 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --encoder_name llama3b \
    --decoder_name llama3b \
    --untrainable_nbit 3.6 \
    --wandb

# 3bto3b aug0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new3/train.py \
    --tag 0116_3bto3b_aug0.3 \
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

# Submitted batch job 55973905
# Submitted batch job 55973906
# Submitted batch job 55974655
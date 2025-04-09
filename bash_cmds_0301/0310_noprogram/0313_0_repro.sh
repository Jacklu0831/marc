# in past runs, noprogram bos seems worse than base and llama3b is similar to llama1b maybe due to quantization, need to double check

# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0310_noprogram/0313_0_repro.sh
# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0310_noprogram/0313_0_repro.sh

# noprogram base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_base \
    --wandb

# noprogram quantized
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_quantized \
    --untrainable_nbit 4 \
    --wandb

# noprogram nobos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nobos \
    --no_bos \
    --wandb

# noprogram llama3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_llama3b \
    --model_name llama3b \
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --untrainable_nbit 4 \
    --wandb

# Submitted batch job 58264483
# Submitted batch job 58264484
# Submitted batch job 58264485
# Submitted batch job 58264486 # llama3 with 4gpu
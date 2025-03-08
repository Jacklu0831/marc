# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0222_noprogram/0301_0_llama3b.sh

# noprogram base llama3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_llama3b \
    --model_name llama3b \
    --wandb


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag test \
    --model_name llama3b
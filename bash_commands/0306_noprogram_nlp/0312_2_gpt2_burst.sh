# python make_sbatch.py --ngpu 2 --time 12 --gb 64 --bash_files bash_commands/0306_noprogram_nlp/0312_2_gpt2_burst.sh --burst

# noprogram nlp gpt2 nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0312_noprogram_nlp_gpt2_nolora_burst \
    --model_name gpt2 \
    --no_lora \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --eval_pretrained \
    --wandb

# noprogram nlp llama1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0312_noprogram_nlp_llama1b_burst \
    --model_name llama1b \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --eval_pretrained \
    --wandb

# noprogram nlp llama1b nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0312_noprogram_nlp_llama1b_nolora_burst \
    --model_name llama1b \
    --no_lora \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --eval_pretrained \
    --wandb

# Submitted batch job 37664
# Submitted batch job 37665
# Submitted batch job 37666
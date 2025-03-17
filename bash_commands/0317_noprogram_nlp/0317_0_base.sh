# python make_sbatch.py --ngpu 2 --time 48 --gb 64 --bash_files bash_commands/0317_noprogram_nlp/0317_0_base.sh --burst
# python make_sbatch.py --ngpu 4 --time 48 --gb 32 --bash_files bash_commands/0317_noprogram_nlp/0317_0_base.sh --burst

# noprogram nlp gpt2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0317_noprogram_nlp_gpt2 \
    --model_name gpt2 \
    --no_lora \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --eval_pretrained \
    --wandb

# noprogram nlp gpt2 lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0317_noprogram_nlp_gpt2_lora \
    --model_name gpt2 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --eval_pretrained \
    --wandb

# noprogram nlp llama1b lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0317_noprogram_nlp_llama1b_lora \
    --model_name llama1b \
    --max_seq_len 8192 \
    --max_pair_len 2048 \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --grad_accum_steps 8 \
    --samples_per_epoch 20000 \
    --eval_epochs 2 \
    --eval_pretrained \
    --wandb

# noprogram nlp llama1b lora origscheme
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0317_noprogram_nlp_llama1b_lora_origscheme \
    --model_name llama1b \
    --max_seq_len 8192 \
    --max_pair_len 2048 \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --grad_accum_steps 2 \
    --eval_pretrained \
    --wandb

# Submitted batch job 38660
# Submitted batch job 38661
# Submitted batch job 38662
# Submitted batch job 38735
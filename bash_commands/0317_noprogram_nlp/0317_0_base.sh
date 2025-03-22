# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0317_noprogram_nlp/0317_0_base.sh

# noprogram nlp llama1b lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0317_noprogram_nlp_llama1b_lora \
    --model_name llama1b \
    --max_seq_len 8192 \
    --max_pair_len 2048 \
    --train_batch_size 2 \
    --grad_accum_steps 4 \
    --eval_batch_size 4 \
    --eval_pretrained \
    --wandb

# Submitted batch job 58496820 # old gpt2 stuff
# Submitted batch job 58496821 # old gpt2 stuff
# Submitted batch job 58496822
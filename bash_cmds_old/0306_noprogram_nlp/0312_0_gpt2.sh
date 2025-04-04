# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0306_noprogram_nlp/0312_0_gpt2.sh

# noprogram nlp gpt2 nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_gpt2_nolora \
    --model_name gpt2 \
    --no_lora \
    --wandb

# noprogram nlp llama1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b \
    --model_name llama1b \
    --wandb

# noprogram nlp llama1b excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_excludefirst \
    --model_name llama1b \
    --loss_type exclude_first \
    --wandb

# noprogram nlp llama1b evalminpair2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_evalminpair2 \
    --model_name llama1b \
    --eval_min_num_pair 2 \
    --wandb

# noprogram nlp llama1b excludefirst evalminpair2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_excludefirst_evalminpair2 \
    --model_name llama1b \
    --loss_type exclude_first \
    --eval_min_num_pair 2 \
    --wandb

# noprogram nlp llama1b nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_nolora \
    --model_name llama1b \
    --no_lora \
    --wandb

# Submitted batch job 58209647
# Submitted batch job 58209648
# Submitted batch job 58209649
# Submitted batch job 58209650
# Submitted batch job 58209651
# Submitted batch job 58209652
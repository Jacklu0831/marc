# python make_sbatch.py --ngpu 2 --time 12 --bash_files bash_commands/0306_noprogram_nlp/0312_3_gpt2_big.sh

# noprogram nlp llama1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0312_noprogram_nlp_llama1b_burst \
    --model_name llama1b \
    --max_seq_len 8192 \
    --max_pair_len 1024 \
    --train_batch_size 2 \
    --eval_batch_size 16 \
    --eval_pretrained \
    --wandb


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --debug_no_resume \
    --tag test \
    --model_name llama1b \
    --max_seq_len 8192 \
    --max_pair_len 1024 \
    --train_batch_size 2 \
    --grad_accum_steps 8 \
    --eval_batch_size 4 \
    --eval_pretrained \
    --eval_train_ratio 1e-4 \
    --eval_eval_ratio 1e-4 \
    --num_epochs 40
# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0306_noprogram_nlp/0313_1_llama_big.sh

# noprogram nlp llama1b big
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_big \
    --model_name llama1b \
    --max_seq_len 8192 \
    --max_pair_len 2048 \
    --train_batch_size 2 \
    --grad_accum_steps 8 \
    --eval_batch_size 4 \
    --samples_per_epoch 20000 \
    --eval_epochs 2 \
    --eval_pretrained \
    --wandb

# Submitted batch job 58263627
# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0222_noprogram/0309_0_llama3b.sh

# noprogram llama3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0309_noprogram_llama3b \
    --model_name llama3b \
    --grad_accum_steps 8 \
    --train_batch_size 1 \
    --untrainable_nbit 4 \
    --eval_batch_size 8 \
    --wandb


# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
#     --lr_scheduler constant \
#     --tag test \
#     --model_name llama3b \
#     --grad_accum_steps 8 \
#     --train_batch_size 1 \
#     --untrainable_nbit 4 \
#     --eval_batch_size 8 \
#     --log_every 1 \
#     --debug_no_resume \
#     --num_epoch 40 \
#     --debug_len 8192



# Submitted batch job 58107133
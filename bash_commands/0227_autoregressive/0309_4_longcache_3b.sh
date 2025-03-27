# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0227_autoregressive/0309_4_longcache_3b.sh
# even 4 days might not be enough... 40 epochs take 300-350hrs on single gpu

# arlongcache llama3b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_llama3b \
    --model_name llama3b \
    --grad_accum_steps 8 \
    --train_batch_size 1 \
    --untrainable_nbit 4 \
    --eval_batch_size 2 \
    --wandb

# # check memory, seems like archeckpointing is not required???
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
#     --lr_scheduler constant \
#     --token_weighted_loss \
#     --tag test \
#     --model_name llama3b \
#     --grad_accum_steps 8 \
#     --train_batch_size 1 \
#     --untrainable_nbit 4 \
#     --eval_batch_size 2 \
#     --log_every 1 \
#     --debug_no_resume \
#     --num_epoch 40 \
#     --debug_len 8192 \
#     --debug

# Submitted batch job 58170492
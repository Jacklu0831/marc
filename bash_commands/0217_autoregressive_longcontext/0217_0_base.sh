# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0217_autoregressive_longcontext/0217_0_base.sh

# arlong batchsize4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0217_arlong_batchsize4 \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --grad_accum_steps 2 \
    --gradient_checkpointing \
    --wandb

# Submitted batch job 57358892
# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0127_3_gs.sh

# ar novae gs50
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0204/train.py \
    --tag 0202_ar_novae_gs50 \
    --train_no_sample \
    --eval_no_sample \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 50 \
    --wandb

# Submitted batch job 56923723
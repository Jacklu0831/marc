# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_5_gs.sh
# make sure gs works by applying it to model that works without vae

# single gs50
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_gs50 \
    --eval_batch_size 1 \
    --gs_batch_size 10000 \
    --gs_take_best \
    --gs_iters 50 \
    --wandb

# Submitted batch job 56865817
# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0207_0_debug_gs_evalbatchsize1.sh
# make sure gs works by applying it to model that works without vae

# single debug gs evalbatchsize1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0205/train.py \
    --tag 0207_debug_gs_evalbatchsize1 \
    --eval_batch_size 1 \
    --wandb

# Submitted batch job 57044888
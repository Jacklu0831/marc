# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0327_autoregressive_toy/0327_0_shortnumpair.sh

# arlongcache toy numpair21
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_toy/train.py \
    --lr_scheduler constant \
    --tag 0327_arlongcache_toy_numpair21 \
    --min_train_num_pair 21 \
    --max_train_num_pair 21 \
    --min_eval_num_pair 21 \
    --max_eval_num_pair 21 \
    --train_batch_size 1024 \
    --grad_accum_steps 2 \
    --lr_program 1e-3 \
    --lr_prior 1e-3 \
    --lr_other 1e-3 \
    --wandb

# python make_sbatch.py --rtx8000 --ngpu 1 --time 24 --bash_files bash_commands/0327_noprogram_toy/0327_2_shortnumpair.sh

# noprogram toy numpair21
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy/train.py \
    --lr_scheduler constant \
    --tag 0327_noprogram_toy_numpair21 \
    --min_train_num_pair 21 \
    --max_train_num_pair 21 \
    --min_eval_num_pair 21 \
    --max_eval_num_pair 21 \
    --train_batch_size 2048 \
    --lr_other 1e-3 \
    --wandb

# Submitted batch job 58790384
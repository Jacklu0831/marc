# python make_sbatch.py --rtx8000 --ngpu 1 --time 24 --bash_files bash_commands/0327_noprogram_toy/0327_1_numtrainnet.sh
# lr4e-4, gradnorm1

# noprogram toy numpair101 numtrainnet100
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy/train.py \
    --lr_scheduler constant \
    --tag 0327_noprogram_toy_numpair101_numtrainnet100 \
    --min_train_num_pair 101 \
    --max_train_num_pair 101 \
    --min_eval_num_pair 101 \
    --max_eval_num_pair 101 \
    --num_train_net 100 \
    --wandb

# noprogram toy numpair101 numtrainnet1000
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy/train.py \
    --lr_scheduler constant \
    --tag 0327_noprogram_toy_numpair101_numtrainnet1000 \
    --min_train_num_pair 101 \
    --max_train_num_pair 101 \
    --min_eval_num_pair 101 \
    --max_eval_num_pair 101 \
    --num_train_net 1000 \
    --wandb

# noprogram toy numpair101 numtrainnet10000
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy/train.py \
    --lr_scheduler constant \
    --tag 0327_noprogram_toy_numpair101_numtrainnet10000 \
    --min_train_num_pair 101 \
    --max_train_num_pair 101 \
    --min_eval_num_pair 101 \
    --max_eval_num_pair 101 \
    --num_train_net 10000 \
    --wandb

# noprogram toy numpair101 numtrainnet100000
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy/train.py \
    --lr_scheduler constant \
    --tag 0327_noprogram_toy_numpair101_numtrainnet100000 \
    --min_train_num_pair 101 \
    --max_train_num_pair 101 \
    --min_eval_num_pair 101 \
    --max_eval_num_pair 101 \
    --num_train_net 100000 \
    --wandb

# noprogram toy numpair101
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy/train.py \
    --lr_scheduler constant \
    --tag 0327_noprogram_toy_numpair101 \
    --min_train_num_pair 101 \
    --max_train_num_pair 101 \
    --min_eval_num_pair 101 \
    --max_eval_num_pair 101 \
    --wandb

# Submitted batch job 58790378
# Submitted batch job 58790379
# Submitted batch job 58790380
# Submitted batch job 58790381
# Submitted batch job 58790382
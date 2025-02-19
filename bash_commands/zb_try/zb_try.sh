# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive_longcontext/0211_0_base.sh

# zb_try
accelerate launch --main_process_port 41845 --mixed_precision bf16 encoder_decoder_singleprogram/train.py --tag test


# python make_sbatch.py --ngpu 2 --time 24 --gb 64 --bash_files bash_commands/0306_noprogram_nlp/0313_0_gpt2_burst.sh --burst

# noprogram nlp gpt2 nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_gpt2_nolora_burst \
    --model_name gpt2 \
    --no_lora \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --eval_pretrained \
    --debug_no_resume \
    --wandb

# Submitted batch job 37755 # train acc not doing up, while eval does???????

# Submitted batch job 37945 # only difference being eval on all
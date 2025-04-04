# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0217_autoregressive/0217_1_extrainference.sh

# ar extrainference2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_extrainference2 \
    --eval_batch_size 64 \
    --extra_inference_pairs 2 \
    --wandb

# ar extrainference5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_extrainference5 \
    --eval_batch_size 64 \
    --extra_inference_pairs 5 \
    --wandb

# ar extrainference10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_extrainference10 \
    --eval_batch_size 64 \
    --extra_inference_pairs 10 \
    --wandb

# Submitted batch job 57338607
# Submitted batch job 57338608
# Submitted batch job 57338609
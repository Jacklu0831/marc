# python make_sbatch.py --ngpu 1 --time 48 --gb 64 --bash_files bash_cmds/0401_nlp/0401_0_base.sh
# try a lora without embeddings so we can properly merge it

# nlp gpt2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/train.py \
    --lr_scheduler constant \
    --tag 0401_nlp_gpt2 \
    --eval_pretrained \
    --allow_truncate \
    --wandb

# nlp gpt2 notruncate
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/train.py \
    --lr_scheduler constant \
    --tag 0401_nlp_gpt2_notruncate \
    --eval_pretrained \
    --wandb

# nlp gpt2 newline
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/train.py \
    --lr_scheduler constant \
    --tag 0401_nlp_gpt2_newline \
    --eval_pretrained \
    --allow_truncate \
    --delimiter newline \
    --wandb

# Submitted batch job 59006197 # middle
# Submitted batch job 59006198 # best
# Submitted batch job 59006199 # worst

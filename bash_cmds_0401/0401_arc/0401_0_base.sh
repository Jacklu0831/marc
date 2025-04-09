# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_cmds/0401_arc/0401_0_base.sh
# try a lora without embeddings so we can properly merge it

# arc
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0401_arc \
    --wandb

# Submitted batch job 59029872
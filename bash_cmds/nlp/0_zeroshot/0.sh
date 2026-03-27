# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --bash_files bash_cmds/nlp/0_zeroshot/0.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp zeroshot
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_zeroshot \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --zero_shot

# old:
# 0.356599459564066

# new:
# Submitted batch job 64197562 # killed, annoying
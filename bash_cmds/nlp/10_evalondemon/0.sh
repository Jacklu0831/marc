# python make_sbatch.py --ngpu 1 --time 1 --rtx8000 --single --bash_files bash_cmds/nlp/10_evalondemon/0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_evalondemon \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --eval_on_demonstrations

# old run got 0.8086 on evalseed100, <3min

# Submitted batch job 60167583
# 0.8191903628174325
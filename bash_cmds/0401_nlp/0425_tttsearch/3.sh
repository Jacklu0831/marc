# python make_sbatch.py --ngpu 1 --time 32 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_tttsearch/3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp ttt iter250
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter250 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 1600

# nlp ttt iter200
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter200 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 200 \
    --ttt_permute_n 1600

# Submitted batch job 59760715
# Submitted batch job 59817841

# 0.4422030355473379
# 0.4444554967916575
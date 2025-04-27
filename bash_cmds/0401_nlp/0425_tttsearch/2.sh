# python make_sbatch.py --ngpu 1 --time 32 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_tttsearch/2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp ttt iter300
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter300 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 300 \
    --ttt_permute_n 1600

# nlp ttt iter150
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter150 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 150 \
    --ttt_permute_n 1600

# Submitted batch job 59760714

# 0.4423190101467986
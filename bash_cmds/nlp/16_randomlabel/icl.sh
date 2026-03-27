# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/nlp/16_randomlabel/icl.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp icl wronglabel0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_icl_wronglabel0.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --wrong_label 0.0

# nlp icl wronglabel0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_icl_wronglabel0.25 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --wrong_label 0.25

# nlp icl wronglabel0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_icl_wronglabel0.5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --wrong_label 0.5

# nlp icl wronglabel0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_icl_wronglabel0.75 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --wrong_label 0.75

# nlp icl wronglabel1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_icl_wronglabel1.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --wrong_label 1.0

# Submitted batch job 64234606
# Submitted batch job 64234607
# Submitted batch job 64234608
# Submitted batch job 64234609
# Submitted batch job 64234610

# 0.3570521274621764
# 0.36103693129438497
# 0.35367271395911926
# 0.3471519021212006
# 0.3170646833601794
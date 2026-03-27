# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --bash_files bash_cmds/nlp/16_randomlabel/ctkv_lr1e-3.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp gs50 lr1e-3 tokendrop0.05 wronglabel0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_wronglabel0.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.0

# nlp gs100 lr1e-3 tokendrop0.05 wronglabel0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_wronglabel0.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.0

# nlp gs150 lr1e-3 tokendrop0.05 wronglabel0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_wronglabel0.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.0

# nlp gs200 lr1e-3 tokendrop0.05 wronglabel0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_wronglabel0.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.0







# nlp gs50 lr1e-3 tokendrop0.05 wronglabel0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_wronglabel0.25 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.25

# nlp gs100 lr1e-3 tokendrop0.05 wronglabel0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_wronglabel0.25 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.25

# nlp gs150 lr1e-3 tokendrop0.05 wronglabel0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_wronglabel0.25 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.25

# nlp gs200 lr1e-3 tokendrop0.05 wronglabel0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_wronglabel0.25 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.25








# nlp gs50 lr1e-3 tokendrop0.05 wronglabel0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_wronglabel0.5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.5

# nlp gs100 lr1e-3 tokendrop0.05 wronglabel0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_wronglabel0.5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.5

# nlp gs150 lr1e-3 tokendrop0.05 wronglabel0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_wronglabel0.5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.5

# nlp gs200 lr1e-3 tokendrop0.05 wronglabel0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_wronglabel0.5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.5






# nlp gs50 lr1e-3 tokendrop0.05 wronglabel0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_wronglabel0.75 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.75

# nlp gs100 lr1e-3 tokendrop0.05 wronglabel0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_wronglabel0.75 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.75

# nlp gs150 lr1e-3 tokendrop0.05 wronglabel0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_wronglabel0.75 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.75

# nlp gs200 lr1e-3 tokendrop0.05 wronglabel0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_wronglabel0.75 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 0.75







# nlp gs50 lr1e-3 tokendrop0.05 wronglabel1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_wronglabel1.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 1.0

# nlp gs100 lr1e-3 tokendrop0.05 wronglabel1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_wronglabel1.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 1.0

# nlp gs150 lr1e-3 tokendrop0.05 wronglabel1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_wronglabel1.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 1.0

# nlp gs200 lr1e-3 tokendrop0.05 wronglabel1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_wronglabel1.0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --wrong_label 1.0


# Submitted batch job 64234585
# Submitted batch job 64234586
# Submitted batch job 64234587
# Submitted batch job 64234588

# Submitted batch job 64234589
# Submitted batch job 64234590
# Submitted batch job 64234591
# Submitted batch job 64234592

# Submitted batch job 64234593
# Submitted batch job 64234594
# Submitted batch job 64234595
# Submitted batch job 64234596

# Submitted batch job 64234597
# Submitted batch job 64234598
# Submitted batch job 64234599
# Submitted batch job 64234600

# Submitted batch job 64234601
# Submitted batch job 64234602
# Submitted batch job 64234603
# Submitted batch job 64234604
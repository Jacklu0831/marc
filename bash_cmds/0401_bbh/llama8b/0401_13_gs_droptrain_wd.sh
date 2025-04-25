# python make_sbatch.py --ngpu 1 --time 10 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_13_gs_droptrain_wd.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# bbh llama8b gs10 lr1e-3 droptrain wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_wd0.01 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.01

# bbh llama8b gs20 lr1e-3 droptrain wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_wd0.01 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.01

# bbh llama8b gs30 lr1e-3 droptrain wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_wd0.01 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.01

# bbh llama8b gs40 lr1e-3 droptrain wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_wd0.01 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.01

# bbh llama8b gs50 lr1e-3 droptrain wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_wd0.01 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.01












# bbh llama8b gs10 lr1e-3 droptrain wd0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_wd0.05 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.05

# bbh llama8b gs20 lr1e-3 droptrain wd0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_wd0.05 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.05

# bbh llama8b gs30 lr1e-3 droptrain wd0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_wd0.05 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.05

# bbh llama8b gs40 lr1e-3 droptrain wd0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_wd0.05 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.05

# bbh llama8b gs50 lr1e-3 droptrain wd0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_wd0.05 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.05










# bbh llama8b gs10 lr1e-3 droptrain wd0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_wd0.1 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.1

# bbh llama8b gs20 lr1e-3 droptrain wd0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_wd0.1 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.1

# bbh llama8b gs30 lr1e-3 droptrain wd0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_wd0.1 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.1

# bbh llama8b gs40 lr1e-3 droptrain wd0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_wd0.1 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.1

# bbh llama8b gs50 lr1e-3 droptrain wd0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_wd0.1 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_weight_decay 0.1

# Submitted batch job 59473326

# wd0.01
# 52.907582367535845
# 53.14092554787136 <-
# 53.10839307474403
# 51.66673260641356
# 50.97084408151208

# wd0.05
# 52.659622543304835
# 53.31969039970036 <---
# 53.00447203363454
# 51.612738943662826
# 50.641888268176075

# wd0.03
# 53.16548150961579 <-
# 52.73741166271906
# 53.05240803163702
# 51.59544514604335
# 51.088706983107116

# so far 53.31969039970036
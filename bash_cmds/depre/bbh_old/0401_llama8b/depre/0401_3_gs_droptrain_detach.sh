# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_3_gs_droptrain_detach.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)






# bbh llama8b gs10 lr1e-3 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach

# bbh llama8b gs20 lr1e-3 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach

# bbh llama8b gs30 lr1e-3 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach

# bbh llama8b gs40 lr1e-3 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach

# bbh llama8b gs50 lr1e-3 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach










# bbh llama8b gs10 lr1e-4 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-4_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach

# bbh llama8b gs20 lr1e-4 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-4_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach

# bbh llama8b gs30 lr1e-4 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-4_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach

# bbh llama8b gs40 lr1e-4 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-4_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach

# bbh llama8b gs50 lr1e-4 droptrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-4_droptrain_detach \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_detach

# Submitted batch job 59408785

# lr1e-3
# 48.0858109094234
# 47.604600220502505
# 47.385568163235156
# 46.67957327312595
# 46.44259901072797

# lr1e-4
# 49.183582409737284
# 48.82980247968616
# 49.07848544314147
# 48.61928324374318
# 48.50205995769305

# so far 49.07848544314147
# <5hrs
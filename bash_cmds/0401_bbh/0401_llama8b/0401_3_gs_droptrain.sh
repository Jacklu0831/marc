# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_3_gs_droptrain.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)







# bbh llama8b gs10 lr3e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs20 lr3e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs30 lr3e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs40 lr3e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs50 lr3e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train








# bbh llama8b gs10 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs20 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs30 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs40 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs50 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train










# bbh llama8b gs10 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-4_droptrain \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs20 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-4_droptrain \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs30 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-4_droptrain \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs40 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-4_droptrain \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gs50 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-4_droptrain \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train

# Submitted batch job 59411913

# lr3e-3 (too high)
# 51.62924586030268 <-
# 48.107790825055694
# 47.80690116599056
# 45.736175512263905
# 45.194878327979026

# lr1e-3
# 52.720192596812744
# 53.22709780710778 <---
# 53.00635351441268
# 51.454463769186276
# 51.38944717874593

# lr1e-4
# 49.66758234995189
# 50.45712949335415
# 50.87928894093733
# 51.964760480463376
# 52.62106097931954 <-

# so far 53.22709780710778
# <3hrs

# times
# 16
# 18
# 20
# 24
# 27
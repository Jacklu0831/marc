# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_3_gs_droptrain_ntoken32.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)






# bbh llama8b gs10 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-1_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs20 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-1_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs30 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-1_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs40 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-1_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs50 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-1_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-1 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32









# bbh llama8b gs10 lr1e-2 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-2_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs20 lr1e-2 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-2_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs30 lr1e-2 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-2_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs40 lr1e-2 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-2_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs50 lr1e-2 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-2_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32










# bbh llama8b gs10 lr1e-3 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs20 lr1e-3 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs30 lr1e-3 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs40 lr1e-3 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# bbh llama8b gs50 lr1e-3 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_ntoken32 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_ntokens 32

# Submitted batch job 59473316

# lr1e-1 (too high)
# 25.819962950654215
# 27.436794554607783
# 23.81515945989193
# 27.14528812152608
# 25.947281611954256

# lr1e-2
# 47.82806123076976
# 48.75093634440592
# 48.804393697215225 <-
# 47.648450152188936
# 47.22049240286195

# lr1e-3
# 37.280284367356494
# 38.660919797925445
# 39.709913469468134
# 43.04490628643171
# 45.32341907258824 <-

# so far 48.804393697215225
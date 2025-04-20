# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_2_gs.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)






# bbh llama8b gs10 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b gs20 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b gs30 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b gs40 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b gs50 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2










# bbh llama8b gs10 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b gs20 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b gs30 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b gs40 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b gs50 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# Submitted batch job 59411912

# lr1e-3
# 48.83875050334006 <-
# 48.147233783657846
# 47.67402598201867
# 47.306721809879
# 47.467823601505884

# lr1e-4
# 49.3453326088753 <-
# 49.02523250354756
# 49.02160142148512
# 48.87501516614179
# 48.72506598370672

# so far 49.3453326088753
# <3hrs
# python make_sbatch.py --ngpu 1 --time 5 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_3_gs_droptrain_tokendropout0.01.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)






# bbh llama8b gs10 lr1e-3 droptrain tokendropout0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_tokendropout0.01 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_dropout 0.01

# bbh llama8b gs20 lr1e-3 droptrain tokendropout0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_tokendropout0.01 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_dropout 0.01

# bbh llama8b gs30 lr1e-3 droptrain tokendropout0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_tokendropout0.01 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_dropout 0.01

# bbh llama8b gs40 lr1e-3 droptrain tokendropout0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_tokendropout0.01 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_dropout 0.01

# bbh llama8b gs50 lr1e-3 droptrain tokendropout0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_tokendropout0.01 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_token_dropout 0.01

# beat 53.22709780710778

# 52.35677886973998
# 53.01040001688056
# 53.32370173430326 <-
# 52.17325097019348
# 51.721198837350386

# so far 53.32370173430326
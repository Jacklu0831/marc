# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_bbh/llama8b/0419_3_llama1b_gs_droptrain_fullprecision.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)






# bbh llama1b gs10 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs10_lr1e-3_droptrain \
    --model_name llama1b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama1b gs20 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs20_lr1e-3_droptrain \
    --model_name llama1b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama1b gs30 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs30_lr1e-3_droptrain \
    --model_name llama1b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama1b gs40 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs40_lr1e-3_droptrain \
    --model_name llama1b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama1b gs50 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs50_lr1e-3_droptrain \
    --model_name llama1b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train










# bbh llama1b gs10 lr1e-3 droptrain fullprecision
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag bbh_llama1b_gs10_lr1e-3_droptrain_fullprecision \
    --model_name llama1b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama1b gs20 lr1e-3 droptrain fullprecision
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag bbh_llama1b_gs20_lr1e-3_droptrain_fullprecision \
    --model_name llama1b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama1b gs30 lr1e-3 droptrain fullprecision
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag bbh_llama1b_gs30_lr1e-3_droptrain_fullprecision \
    --model_name llama1b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama1b gs40 lr1e-3 droptrain fullprecision
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag bbh_llama1b_gs40_lr1e-3_droptrain_fullprecision \
    --model_name llama1b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama1b gs50 lr1e-3 droptrain fullprecision
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag bbh_llama1b_gs50_lr1e-3_droptrain_fullprecision \
    --model_name llama1b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# Submitted batch job 59479554
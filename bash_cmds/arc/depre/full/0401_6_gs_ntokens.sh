# python make_sbatch.py --ngpu 1 --time 8 --bash_files bash_cmds/0401_arc/full/0401_6_gs_ntokens.sh





# arc gs5 lr1e-1 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-1_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-1 \
    --gs_ntokens 32

# arc gs25 lr1e-1 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-1_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-1 \
    --gs_ntokens 32

# arc gs100 lr1e-1 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-1_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-1 \
    --gs_ntokens 32

# arc gs250 lr1e-1 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-1_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-1 \
    --gs_ntokens 32







# arc gs5 lr1e-2 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-2_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_ntokens 32

# arc gs25 lr1e-2 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-2_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_ntokens 32

# arc gs100 lr1e-2 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-2_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_ntokens 32

# arc gs250 lr1e-2 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-2_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_ntokens 32







# arc gs5 lr1e-3 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-3_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_ntokens 32

# arc gs25 lr1e-3 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-3_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_ntokens 32

# arc gs100 lr1e-3 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-3_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_ntokens 32

# arc gs250 lr1e-3 ntokens32 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-3_ntokens32_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_ntokens 32






# Submitted batch job 59293641 # 0.145
# Submitted batch job 59293642 # 0.1975
# Submitted batch job 59293643 # 0.175
# Submitted batch job 59293644 # 0.1875

# Submitted batch job 59293645 # 0.135
# Submitted batch job 59293646 # 0.1375
# Submitted batch job 59293647 # 0.17
# Submitted batch job 59293648 # 0.2025

# Submitted batch job 59293649 # 0.1325
# Submitted batch job 59293650 # 0.135
# Submitted batch job 59293651 # 0.1375
# Submitted batch job 59293652 # 0.14

# so far 0.2025
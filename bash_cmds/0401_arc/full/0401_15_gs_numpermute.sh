# python make_sbatch.py --ngpu 1 --time 16 --bash_files bash_cmds/0401_arc/full/0401_15_gs_numpermute.sh





# arc gs5 lr3e-2 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr3e-2_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 3e-2 \
    --gs_num_permute 1024

# arc gs25 lr3e-2 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr3e-2_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 3e-2 \
    --gs_num_permute 1024

# arc gs100 lr3e-2 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr3e-2_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 3e-2 \
    --gs_num_permute 1024

# arc gs250 lr3e-2 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr3e-2_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 3e-2 \
    --gs_num_permute 1024








# arc gs5 lr1e-2 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-2_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024

# arc gs25 lr1e-2 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-2_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024

# arc gs100 lr1e-2 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-2_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024

# arc gs250 lr1e-2 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-2_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024









# arc gs5 lr3e-3 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr3e-3_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 3e-3 \
    --gs_num_permute 1024

# arc gs25 lr3e-3 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr3e-3_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 3e-3 \
    --gs_num_permute 1024

# arc gs100 lr3e-3 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr3e-3_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 3e-3 \
    --gs_num_permute 1024

# arc gs250 lr3e-3 permuten1024 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr3e-3_permuten1024_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 3e-3 \
    --gs_num_permute 1024


# lr3e-2
# Submitted batch job 59173396 # 0.1175
# Submitted batch job 59173397 # 0.1125
# Submitted batch job 59173398 # 0.09
# Submitted batch job 59173399 # 0.085

# lr1e-2
# Submitted batch job 59173400 # 0.1275
# Submitted batch job 59173401 # 0.1575
# Submitted batch job 59173402 # 0.1575
# Submitted batch job 59173403 # 0.155

# lr3e-3
# Submitted batch job 59173404 # 0.09
# Submitted batch job 59173405 # 0.1475
# Submitted batch job 59173406 # 0.175
# Submitted batch job 59173407 # 0.185

# so far 0.185
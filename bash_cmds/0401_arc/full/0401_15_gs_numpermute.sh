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



# AFTER PRECISION FIX

# Submitted batch job 59139640
# Submitted batch job 59139641
# Submitted batch job 59139642
# Submitted batch job 59139643

# Submitted batch job 59139644
# Submitted batch job 59139645
# Submitted batch job 59139646
# Submitted batch job 59139647

# Submitted batch job 59139648
# Submitted batch job 59139649
# Submitted batch job 59139650
# Submitted batch job 59139651
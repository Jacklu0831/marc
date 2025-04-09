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
# Submitted batch job 59130002
# Submitted batch job 59130003
# Submitted batch job 59130004
# Submitted batch job 59130005

# lr1e-2
# Submitted batch job 59130006
# Submitted batch job 59130007
# Submitted batch job 59130008
# Submitted batch job 59130009

# lr3e-3
# Submitted batch job 59130010
# Submitted batch job 59130011
# Submitted batch job 59130012
# Submitted batch job 59130013
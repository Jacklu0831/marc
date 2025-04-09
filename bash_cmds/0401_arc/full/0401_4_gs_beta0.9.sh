# python make_sbatch.py --ngpu 1 --time 8 --bash_files bash_cmds/0401_arc/full/0401_4_gs_beta0.9.sh


# arc gs5 lr1e-3 beta0.9 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-3_beta0.9_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9

# arc gs25 lr1e-3 beta0.9 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-3_beta0.9_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9

# arc gs100 lr1e-3 beta0.9 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-3_beta0.9_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9

# arc gs250 lr1e-3 beta0.9 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-3_beta0.9_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9





# arc gs5 lr1e-4 beta0.9 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-4_beta0.9_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9

# arc gs25 lr1e-4 beta0.9 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-4_beta0.9_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9

# arc gs100 lr1e-4 beta0.9 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-4_beta0.9_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9

# arc gs250 lr1e-4 beta0.9 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-4_beta0.9_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9


# AFTER PRECISION FIX

# lr1e-3
# Submitted batch job 59139585
# Submitted batch job 59139586
# Submitted batch job 59139587
# Submitted batch job 59139588

# lr1e-4
# Submitted batch job 59139589
# Submitted batch job 59139590
# Submitted batch job 59139591
# Submitted batch job 59139592
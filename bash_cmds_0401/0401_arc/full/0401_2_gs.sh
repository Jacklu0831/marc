# python make_sbatch.py --ngpu 1 --time 8 --bash_files bash_cmds/0401_arc/full/0401_2_gs.sh
# should give similar performance to the semi-successful experiments from before
# if not, compare model loss with original encoder_decoder_noprogram/evaluate.py
# for a 1:1 reproduction, gs_take_best and gs_max_grad_norm=1e8


# arc gs5 lr1e-3 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-3_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3

# arc gs25 lr1e-3 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-3_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3

# arc gs100 lr1e-3 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-3_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3

# arc gs250 lr1e-3 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-3_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3

# Submitted batch job 59047847 # 0.15
# Submitted batch job 59047848 # 0.17
# Submitted batch job 59047849 # 0.195
# Submitted batch job 59047850 # 0.195 # 5 hours

# so far 0.195
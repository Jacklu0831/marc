# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_cmds/0401_arc/full/0402_0_ttt.sh

# arc ttt5 maxpermute full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt5_maxpermute_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 5 \
    --ttt_permute_n 2000

# arc ttt25 maxpermute full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt25_maxpermute_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 25 \
    --ttt_permute_n 2000

# arc ttt100 maxpermute full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt100_maxpermute_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 100 \
    --ttt_permute_n 2000

# arc ttt250 maxpermute full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt250_maxpermute_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 2000

# arc ttt500 maxpermute full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt500_maxpermute_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 500 \
    --ttt_permute_n 2000


# Submitted batch job 59219502 # 0.1425
# Submitted batch job 59219503 # 0.1925
# Submitted batch job 59219504 # 0.2425
# Submitted batch job 59219505 # 0.2375
# Submitted batch job 59219506 # 0.2375
# so far 0.2425
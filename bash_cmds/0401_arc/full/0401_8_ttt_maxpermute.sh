# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_cmds/0401_arc/full/0401_8_ttt_maxpermute.sh



# arc ttt5 permuten1000 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt5_permuten1000_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 5 \
    --ttt_permute_n 1000

# arc ttt25 permuten1000 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt25_permuten1000_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 25 \
    --ttt_permute_n 1000

# arc ttt100 permuten1000 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt100_permuten1000_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 100 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt500 permuten1000 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt500_permuten1000_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 500 \
    --ttt_permute_n 1000


# Submitted batch job 59047912 # 0.14
# Submitted batch job 59047913 # 0.1925
# Submitted batch job 59047914 # 0.245
# Submitted batch job 59047915 # 0.2475 # 8-9hours
# Submitted batch job 59047916 # 0.2375

# so far 0.2475



# AFTER PRECISION FIX

# Submitted batch job 59139611
# Submitted batch job 59139612
# Submitted batch job 59139613
# Submitted batch job 59139614
# Submitted batch job 59139615
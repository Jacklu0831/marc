# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_cmds/0401_arc/full/0401_9_ttt_gs.sh



# arc ttt250 permuten1000 gs5 lr1e-3 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs5_lr1e-3_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs25 lr1e-3 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs25_lr1e-3_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs100 lr1e-3 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs100_lr1e-3_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs250 lr1e-3 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs250_lr1e-3_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000








# arc ttt250 permuten1000 gs5 lr1e-4 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs5_lr1e-4_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs25 lr1e-4 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs25_lr1e-4_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs100 lr1e-4 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs100_lr1e-4_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs250 lr1e-4 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs250_lr1e-4_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000





# arc ttt250 permuten1000 gs5 lr1e-5 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs5_lr1e-5_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs25 lr1e-5 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs25_lr1e-5_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs100 lr1e-5 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs100_lr1e-5_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs250 lr1e-5 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag ttt250_permuten1000_gs250_lr1e-5_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000



# AFTER PRECISION FIX

# lr1e-3
# Submitted batch job 59139618
# Submitted batch job 59139619
# Submitted batch job 59139620
# Submitted batch job 59139621

# lr1e-4
# Submitted batch job 59139622
# Submitted batch job 59139623
# Submitted batch job 59139624
# Submitted batch job 59139625

# lr1e-5
# Submitted batch job 59139626
# Submitted batch job 59139627
# Submitted batch job 59139628
# Submitted batch job 59139629
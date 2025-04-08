# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_cmds/0401_arc/oldfinetuned/full/0401_9_ttt_gs.sh



# arc ttt250 permuten1000 gs5 lr1e-3 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
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
    --flash_attn \
    --tag ttt250_permuten1000_gs250_lr1e-5_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000




# Submitted batch job 59126379
# Submitted batch job 59126380
# Submitted batch job 59126381
# Submitted batch job 59126382

# Submitted batch job 59126383
# Submitted batch job 59126384
# Submitted batch job 59126385
# Submitted batch job 59126386

# Submitted batch job 59126387
# Submitted batch job 59126388
# Submitted batch job 59126389
# Submitted batch job 59126390
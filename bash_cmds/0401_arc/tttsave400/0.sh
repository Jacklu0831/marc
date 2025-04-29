# python make_sbatch.py --ngpu 1 --time 12 --bash_files bash_cmds/0401_arc/tttsave400/0.sh

# arc ttt iter250 save seed0 400task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt_iter250_save_seed0_400task \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 0

# arc ttt iter250 save seed1 400task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt_iter250_save_seed1_400task \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 1

# arc ttt iter250 save seed2 400task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt_iter250_save_seed2_400task \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 2

# arc ttt iter250 save seed3 400task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt_iter250_save_seed3_400task \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 3

# arc ttt iter250 save seed4 400task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_ttt_iter250_save_seed4_400task \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 4

# Submitted batch job 59850193
# Submitted batch job 59850194
# Submitted batch job 59850195
# Submitted batch job 59850196
# Submitted batch job 59850197
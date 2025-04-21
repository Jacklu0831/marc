# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_cmds/0401_arc/small/0401_12_gs_numpermute_permuteback_strip.sh

# arc gs5 lr1e-3 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_numpermute128_permuteback_strip \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --num_permute 128 \
    --permute_back \
    --permute_back_strip_position

# arc gs25 lr1e-3 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3_numpermute128_permuteback_strip \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --num_permute 128 \
    --permute_back \
    --permute_back_strip_position

# arc gs100 lr1e-3 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_numpermute128_permuteback_strip \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --num_permute 128 \
    --permute_back \
    --permute_back_strip_position

# arc gs250 lr1e-3 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_numpermute128_permuteback_strip \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --num_permute 128 \
    --permute_back \
    --permute_back_strip_position





# arc gs5 lr1e-4 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-4_numpermute128_permuteback_strip \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --num_permute 128 \
    --permute_back \
    --permute_back_strip_position

# arc gs25 lr1e-4 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-4_numpermute128_permuteback_strip \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --num_permute 128 \
    --permute_back \
    --permute_back_strip_position

# arc gs100 lr1e-4 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-4_numpermute128_permuteback_strip \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --num_permute 128 \
    --permute_back \
    --permute_back_strip_position

# arc gs250 lr1e-4 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-4_numpermute128_permuteback_strip \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-4 \
    --num_permute 128 \
    --permute_back \
    --permute_back_strip_position

# Submitted batch job 59519233
# Submitted batch job 59519234
# Submitted batch job 59519235
# Submitted batch job 59519236
# Submitted batch job 59519237
# Submitted batch job 59519238
# Submitted batch job 59519239
# Submitted batch job 59519240
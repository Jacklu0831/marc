# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_cmds/0401_arc/small/0401_9_ttt_gs.sh




# arc ttt250 permuten1000 gs5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs5_lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs25_lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs100_lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs250_lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000








# arc ttt250 permuten1000 gs5 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs5_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs25_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs100_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs250 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs250_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000





# arc ttt250 permuten1000 gs5 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs5_lr1e-5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs25 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs25_lr1e-5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs100 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs100_lr1e-5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000

# arc ttt250 permuten1000 gs250 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag ttt250_permuten1000_gs250_lr1e-5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000




# ttt permuten1000 iters250 gets 0.3
# we tune gs on top of it gslr1e-3 and 1e-4 and 1e-5
# result, goes up to 0.3125

# gslr1e-3
# Submitted batch job 59078792 # 0.3
# Submitted batch job 59078793 # 0.3
# Submitted batch job 59078794 # 0.3
# Submitted batch job 59078795 # 0.3

# gslr1e-4
# Submitted batch job 59078796 # 0.3
# Submitted batch job 59078797 # 0.3
# Submitted batch job 59078798 # 0.3
# Submitted batch job 59078799 # 0.3125

# gslr1e-5
# Submitted batch job 59078802 # 0.3
# Submitted batch job 59078803 # 0.3
# Submitted batch job 59078804 # 0.3
# Submitted batch job 59078805 # 0.3








# AFTER PRECISION FIX

# gslr1e-3
# Submitted batch job 59139692
# Submitted batch job 59139693
# Submitted batch job 59139694
# Submitted batch job 59139695

# gslr1e-4
# Submitted batch job 59139696
# Submitted batch job 59139697
# Submitted batch job 59139698
# Submitted batch job 59139699

# gslr1e-5
# Submitted batch job 59139700
# Submitted batch job 59139701
# Submitted batch job 59139702
# Submitted batch job 59139703
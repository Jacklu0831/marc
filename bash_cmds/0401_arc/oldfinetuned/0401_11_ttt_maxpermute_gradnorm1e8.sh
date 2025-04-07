# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_cmds/0401_arc/oldfinetuned/0401_11_ttt_maxpermute_gradnorm1e8.sh



# arc ttt5 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag ttt5_permuten1000_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 5 \
    --ttt_permute_n 1000 \
    --gs_max_grad_norm 1e8

# arc ttt25 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag ttt25_permuten1000_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 25 \
    --ttt_permute_n 1000 \
    --gs_max_grad_norm 1e8

# arc ttt100 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag ttt100_permuten1000_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 100 \
    --ttt_permute_n 1000 \
    --gs_max_grad_norm 1e8

# arc ttt250 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag ttt250_permuten1000_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --gs_max_grad_norm 1e8

# arc ttt500 permuten1000 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag ttt500_permuten1000_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 500 \
    --ttt_permute_n 1000 \
    --gs_max_grad_norm 1e8


# Submitted batch job 59078958
# Submitted batch job 59078959
# Submitted batch job 59078960
# Submitted batch job 59078961
# Submitted batch job 59078962
# python make_sbatch.py --ngpu 1 --time 3 --bash_files bash_cmds/0401_arc/small/0419_1_ttt_ewc_highlr.sh

# arc ttt iter5 lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter5_lambda1e0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 5 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e0

# arc ttt iter25 lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter25_lambda1e0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 25 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e0

# arc ttt iter100 lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter100_lambda1e0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 100 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e0

# arc ttt iter250 lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter250_lambda1e0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e0

# arc ttt iter500 lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter500_lambda1e0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 500 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e0









# arc ttt iter5 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter5_lambda1e-1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 5 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e-1

# arc ttt iter25 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter25_lambda1e-1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 25 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e-1

# arc ttt iter100 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter100_lambda1e-1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 100 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e-1

# arc ttt iter250 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter250_lambda1e-1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e-1

# arc ttt iter500 lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter500_lambda1e-1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 500 \
    --ttt_permute_n 5000 \
    --ttt_lambda_param_sqr 1e-1

# python make_sbatch.py --gb 64 --ngpu 2 --time 48 --bash_files bash_commands/0113_ttt/_0113_1_debug_ttt.sh

# original train lora repro
python test_time_train.py \
    --lora_config configs/ttt/1B_lora_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/test \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected_easy.json \
    --batch_size 2 \
    --epochs 2 \
    --lora_rank 128 \
    --lora_alpha 16.0 \
    --learning_rate 5e-5 \
    --new_format

# get a model
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_1 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 2 \
    --decoder_lm_head \
    --conditioning_method hidden2prompt_full_identity

# 01/14/2025 06:22:15 - INFO - __main__ - Optimizer with 4 embed-params lr=1e-05, 452 other-params lr=0.0001

# get a model
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_2 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 2 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --encoder_lm_head \
    --encoder_loss_lambda 1.0 \
    --tie_models \
    --conditioning_method hidden2prompt_shared

# 01/14/2025 06:23:08 - INFO - __main__ - Optimizer with 2 embed-params lr=0.001, 228 other-params lr=0.01

# get a model
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_3 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 2 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --no_lora \
    --conditioning_method prefix2prefix

# 01/14/2025 06:24:10 - INFO - __main__ - Optimizer with 0 embed-params lr=0.001, 287 other-params lr=0.01

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test \
    --weight_dir test_1 \
    --epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 2 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full_identity \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 10

# 01/14/2025 06:26:50 - INFO - __main__ - saved 226 encoder weights
# 01/14/2025 06:26:50 - INFO - __main__ - saved 228 decoder weights

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test \
    --weight_dir test_2 \
    --epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 2 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_shared \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 10 \
    --tie_models

# 01/14/2025 06:29:56 - INFO - __main__ - saved 228 encoder weights

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test \
    --weight_dir test_3 \
    --epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 2 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method prefix2prefix \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 10 \
    --no_lora



accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test \
    --weight_dir test_3 \
    --epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 1 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method prefix2prefix \
    --encoder_loss_lambda 1.0 \
    --grad_accum_steps 1 \
    --max_samples_per_task 2 \
    --no_lora
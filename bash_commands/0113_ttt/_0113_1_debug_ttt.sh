# python make_sbatch.py --gb 64 --ngpu 2 --time 48 --bash_files bash_commands/0113_ttt/_0113_1_debug_ttt.sh

# original train lora repro
python ttt_old/test_time_train.py \
    --lora_config configs/ttt/1B_lora_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/test \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 1 \
    --lora_rank 128 \
    --lora_alpha 16.0 \
    --learning_rate 5e-5 \
    --new_format

# base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_base \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 2 \
    --conditioning_method hidden2prompt_full_identity

# 01/14/2025 22:28:50 - INFO - __main__ - Optimizer with 4 embed-params lr=1e-05, 450 other-params lr=0.0001

# tiemodels
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_tiemodels \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 2 \
    --tie_models \
    --conditioning_method hidden2prompt_shared

# 01/14/2025 22:28:05 - INFO - __main__ - Optimizer with 2 embed-params lr=1e-05, 226 other-params lr=0.0001

# nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_nolora \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 2 \
    --no_lora \
    --conditioning_method prefix2prefix

# 01/14/2025 22:27:17 - INFO - __main__ - Optimizer with 2 embed-params lr=1e-05, 285 other-params lr=0.0001

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test \
    --weight_dir test_base \
    --weight_epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 2 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full_identity \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 20

# 01/14/2025 21:26:18 - INFO - __main__ - cached 226 lora encoder weights
# 01/14/2025 21:26:18 - INFO - __main__ - cached 226 lora decoder weights
# 01/14/2025 21:26:31 - INFO - __main__ - Optimizer with 4 embed-params lr=1e-05, 450 other-params lr=0.0001

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test_fulllora \
    --weight_dir test_tiemodels \
    --weight_epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 2 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_shared \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 8 \
    --tie_models \
    --full_lora

# 01/14/2025 21:27:36 - INFO - __main__ - cached 226 lora encoder weights
# 01/14/2025 21:27:46 - INFO - __main__ - Optimizer with 2 embed-params lr=1e-05, 226 other-params lr=0.0001

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test_partiallora \
    --weight_dir test_tiemodels \
    --weight_epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 2 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_shared \
    --max_samples_per_task 8 \
    --tie_models

# 01/15/2025 00:36:42 - INFO - __main__ - cached 160 lora encoder weights to ./encoder_decoder/outputs_ttt/ttt_test_partiallora_test_tiemodels/ft_lora_encoder_cache.pt
# 01/15/2025 00:36:47 - INFO - __main__ - Optimizer with 0 embed-params lr=1e-05, 162 other-params lr=0.0001

# DOES NOT WORK, DEPRECATED
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test \
    --weight_dir test_nolora \
    --weight_epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 2 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method prefix2prefix \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 20 \
    --no_lora

# 01/14/2025 22:33:28 - INFO - __main__ - Optimizer with 2 embed-params lr=1e-05, 285 other-params lr=0.0001

# debug ddp error
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/ttt.py \
    --tag test_ddp_error \
    --weight_dir test_tiemodels \
    --weight_epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4_ttt/training \
    --save_epochs 2 \
    --num_epochs 2 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_shared \
    --max_samples_per_task 50 \
    --tie_models \
    --flash_attn \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --full_lora


# train2 ttt fulllora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/ttt.py \
    --tag test2_fulllora \
    --weight_dir test_ttt_2 \
    --weight_epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4_ttt/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method prefix2prefix \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 4 \
    --grad_accum_steps 1 \
    --optimizer sgd \
    --debug_no_aug \
    --max_grad_norm 1e8 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --num_epochs 100 \
    --debug_no_aug \
    --log_every 1 \
    --save_epochs 100 \
    --full_lora
# original lora gets 7/80
python predict.py \
    --experiment_folder inference_outputs/experiments/test \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder train_outputs/1214_0_llama1b_ttt_lora_eval \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --max_lora_rank 128 \
    --include_n 1 \
    --permute_n 2 \
    --new_format \
    --max_tokens 8192
    # --max_tokens 1000000

# no change from before
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original/evaluation \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --train_data_dir ./data/re-arc/train_data/tasks \
    --eval_train_dir ./data/re-arc/arc_original/evaluation \
    --eval_eval_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 2 \
    --grad_accum_steps 1 \
    --compact_grid \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full \
    --no_lora \
    --tie_models \
    --eval_train_select_tasks_path task_info_selected.csv \
    --eval_train_leave_ns 0 \
    --eval_train_permute_n 0 \
    --eval_train_augment_n 0

# mimick predict.py
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original/evaluation \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --train_data_dir ./data/re-arc/train_data/tasks \
    --eval_train_dir ./data/re-arc/arc_original/evaluation \
    --eval_eval_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 2 \
    --grad_accum_steps 1 \
    --compact_grid \
    --conditioning_method hidden2prompt_full \
    --no_lora \
    --tie_models \
    --eval_train_select_tasks_path task_info_selected.csv \
    --eval_train_leave_ns 1 \
    --eval_train_permute_n 2 \
    --eval_train_augment_n 5 \
    --eval_train_leave_ns_inc \
    --max_seq_len 5120
    # --max_seq_len 1000000

# try it overfit1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --num_workers 0 \
    --optimizer sgd \
    --conditioning_method prefix2prefix \
    --eval_train_leave_ns 1 \
    --eval_train_permute_n 1 \
    --eval_train_augment_n 3 \
    --eval_train_leave_ns_inc

# try it overfit4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --num_workers 0 \
    --optimizer sgd \
    --conditioning_method hidden2prefix_full \
    --eval_train_leave_ns 1 \
    --eval_train_permute_n 1 \
    --eval_train_augment_n 3 \
    --eval_train_leave_ns_inc

# eval
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_hidden2prefix_shared \
<<<<<<< HEAD
    --eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prefix_shared \
    --epoch 2 \
    --eval_leave_ns 1 \
    --eval_permute_n 2 \
    --eval_augment_n 5 \
    --eval_leave_ns_inc

# {   'eval/ce_loss': 1.3218018407933414,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}
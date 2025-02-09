# original train lora repro
python test_time_train.py \
    --lora_config configs/ttt/1B_lora_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/test \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --lora_rank 128 \
    --lora_alpha 16.0 \
    --learning_rate 5e-5 \
    --new_format

# train hidden2prompt_full on debugoverfit2_1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_hidden2prompt_debugoverfit2_1 \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit2_1/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
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
    --conditioning_method hidden2prompt_full

# 01/13/2025 05:47:39 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prompt_debugoverfit2_1/encoder_lora_epoch_1
# 01/13/2025 05:47:42 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prompt_debugoverfit2_1/decoder_lora_epoch_1
# 01/13/2025 05:47:42 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_hidden2prompt_debugoverfit2_1/conditioning_projection_epoch_1.pt

# train nolora on debugoverfit2_1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_nolora_debugoverfit2_1 \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit2_1/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
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
    --no_lora

# 01/13/2025 05:44:07 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_nolora_debugoverfit2_1/encoder_lora_epoch_1
# 01/13/2025 05:44:15 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_nolora_debugoverfit2_1/decoder_lora_epoch_1

# train tiemodels on debugoverfit2_1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_tiemodels_debugoverfit2_1 \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit2_1/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
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
    --tie_models

# 01/13/2025 05:44:57 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_tiemodels_debugoverfit2_1/encoder_lora_epoch_2

# train quantized on debugoverfit2_1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_quantized_debugoverfit2_1 \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit2_1/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
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
    --untrainable_nbit 3.6 \
    --trainable_nbit 16

# 01/13/2025 05:46:22 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_quantized_debugoverfit2_1/encoder_lora_epoch_1
# 01/13/2025 05:46:24 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_quantized_debugoverfit2_1/decoder_lora_epoch_1

# train 3bto1b on debugoverfit2_1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_3bto1b_debugoverfit2_1 \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit2_1/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_1/training \
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
    --encoder_name llama3b \
    --decoder_name llama1b \
    --conditioning_method hidden2prompt_full

# 01/13/2025 05:49:22 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_3bto1b_debugoverfit2_1/encoder_lora_epoch_1
# 01/13/2025 05:49:26 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_3bto1b_debugoverfit2_1/decoder_lora_epoch_1
# 01/13/2025 05:49:26 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_3bto1b_debugoverfit2_1/conditioning_projection_epoch_1.pt

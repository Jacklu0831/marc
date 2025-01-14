# training configs:
# conditioning methods
# no lora
# tie models
# 3bto1b
# quantized
# combined

# model size for 1b
# no quantize: 5.4GB, 5.4GB
# quantize: 1.34GB, 1.34GB
# quantize2: 1.74GB, 1.74GB

# prefix2prefix
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_prefix2prefix \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method prefix2prefix

# {   'eval/eval_ce_loss': 0.6310377611246493,
#     'eval/eval_correct_grid_dim': 0.7142857142857143,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.6666666666666667,
#     'eval/eval_valid_grid': 0.7142857142857143,
#     'eval/train_ce_loss': 0.6310377611246493,
#     'eval/train_correct_grid_dim': 0.7142857142857143,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.6666666666666667,
#     'eval/train_valid_grid': 0.7142857142857143}
# 01/12/2025 06:16:00 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_prefix2prefix/eval_train_2.json
# 01/12/2025 06:16:00 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_prefix2prefix/eval_eval_2.json
# 01/12/2025 06:16:07 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_prefix2prefix/encoder_lora_epoch_2
# 01/12/2025 06:16:07 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_prefix2prefix/decoder_lora_epoch_2

# hidden2prefix_shared
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_hidden2prefix_shared \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prefix_shared

# {   'eval/eval_ce_loss': 1.3218018407933414,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 1.3218018407933414,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/12/2025 06:16:51 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_hidden2prefix_shared/eval_train_2.json
# 01/12/2025 06:16:51 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_hidden2prefix_shared/eval_eval_2.json
# 01/12/2025 06:16:58 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prefix_shared/encoder_lora_epoch_2
# 01/12/2025 06:16:58 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prefix_shared/decoder_lora_epoch_2
# 01/12/2025 06:16:59 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_hidden2prefix_shared/conditioning_projection_epoch_2.pt

# hidden2prefix_full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_hidden2prefix_full \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prefix_full

# {   'eval/eval_ce_loss': 1.2037375001569413,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 1.2037375001569413,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/12/2025 06:19:34 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_hidden2prefix_full/eval_train_2.json
# 01/12/2025 06:19:34 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_hidden2prefix_full/eval_eval_2.json
# 01/12/2025 06:19:41 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prefix_full/encoder_lora_epoch_2
# 01/12/2025 06:19:41 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prefix_full/decoder_lora_epoch_2
# 01/12/2025 06:19:42 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_hidden2prefix_full/conditioning_projection_epoch_2.pt

# hidden2prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_hidden2prompt \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt

# {   'eval/eval_ce_loss': 0.2515580686116924,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.8253968253968255,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.2515580686116924,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.8253968253968255,
#     'eval/train_valid_grid': 1.0}
# 01/12/2025 06:20:35 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_hidden2prompt/eval_train_2.json
# 01/12/2025 06:20:35 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_hidden2prompt/eval_eval_2.json
# 01/12/2025 06:20:42 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prompt/encoder_lora_epoch_2
# 01/12/2025 06:20:42 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prompt/decoder_lora_epoch_2

# hidden2prompt_shared
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_hidden2prompt_shared \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt_shared

# {   'eval/eval_ce_loss': 1.0618006943592004,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 1.0618006943592004,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/12/2025 06:28:26 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_hidden2prompt_shared/eval_train_7.json
# 01/12/2025 06:28:26 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_hidden2prompt_shared/eval_eval_7.json
# 01/12/2025 06:28:33 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prompt_shared/encoder_lora_epoch_7
# 01/12/2025 06:28:33 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prompt_shared/decoder_lora_epoch_7
# 01/12/2025 06:28:33 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_hidden2prompt_shared/conditioning_projection_epoch_7.pt

# hidden2prompt_full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_hidden2prompt_full \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt_full

# {   'eval/eval_ce_loss': 0.338758490771787,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.888888888888889,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.338758490771787,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.888888888888889,
#     'eval/train_valid_grid': 1.0}
# 01/12/2025 06:23:41 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_hidden2prompt_full/eval_train_2.json
# 01/12/2025 06:23:41 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_hidden2prompt_full/eval_eval_2.json
# 01/12/2025 06:23:48 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prompt_full/encoder_lora_epoch_2
# 01/12/2025 06:23:48 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prompt_full/decoder_lora_epoch_2
# 01/12/2025 06:23:48 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_hidden2prompt_full/conditioning_projection_epoch_2.pt

# no lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_nolora \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --no_lora

# {   'eval/eval_ce_loss': 0.30878038152669823,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.888888888888889,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.30878038152669823,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.888888888888889,
#     'eval/train_valid_grid': 1.0}
# 01/12/2025 06:42:20 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_nolora/eval_train_2.json
# 01/12/2025 06:42:20 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_nolora/eval_eval_2.json
# 01/12/2025 06:42:27 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_nolora/encoder_lora_epoch_2
# 01/12/2025 06:42:35 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_nolora/decoder_lora_epoch_2

# tie models
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_tiemodels \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --tie_models

# {   'eval/eval_ce_loss': 0.6156056267874581,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 0.6156056267874581,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/12/2025 06:52:31 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_tiemodels/eval_train_12.json
# 01/12/2025 06:52:31 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_tiemodels/eval_eval_12.json
# 01/12/2025 06:52:35 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_tiemodels/encoder_lora_epoch_12

# 3bto1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_3bto1b \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt_full \
    --encoder_name llama3b \
    --decoder_name llama1b

# {   'eval/eval_ce_loss': 0.8358603226287025,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.888888888888889,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.8358603226287025,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.888888888888889,
#     'eval/train_valid_grid': 1.0}
# 01/12/2025 06:46:24 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_3bto1b/eval_train_4.json
# 01/12/2025 06:46:24 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_3bto1b/eval_eval_4.json
# 01/12/2025 06:46:33 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_3bto1b/encoder_lora_epoch_4
# 01/12/2025 06:46:33 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_3bto1b/decoder_lora_epoch_4
# 01/12/2025 06:46:33 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_3bto1b/conditioning_projection_epoch_4.pt

# quantized
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_quantized \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16

# {   'eval/eval_ce_loss': 0.8073542363043609,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.8730158730158731,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.8073542363043609,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.8730158730158731,
#     'eval/train_valid_grid': 1.0}
# 01/12/2025 06:46:28 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_quantized/eval_train_2.json
# 01/12/2025 06:46:28 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_quantized/eval_eval_2.json
# 01/12/2025 06:46:32 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_quantized/encoder_lora_epoch_2
# 01/12/2025 06:46:32 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_quantized/decoder_lora_epoch_2

# quantized2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_quantized2 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 4 \
    --trainable_nbit 32

# {   'eval/eval_ce_loss': 0.7623685662235532,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 0.7623685662235532,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/12/2025 07:02:00 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_quantized2/eval_train_10.json
# 01/12/2025 07:02:00 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_quantized2/eval_eval_10.json
# 01/12/2025 07:02:05 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_quantized2/encoder_lora_epoch_10
# 01/12/2025 07:02:05 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_quantized2/decoder_lora_epoch_10

# quantized3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_quantized3 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 3.6 \
    --trainable_nbit 32

# {   'eval/eval_ce_loss': 0.5667667366755528,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.888888888888889,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.5667667366755528,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.888888888888889,
#     'eval/train_valid_grid': 1.0}
# 01/12/2025 07:42:42 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_quantized3/eval_train_2.json
# 01/12/2025 07:42:42 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_quantized3/eval_eval_2.json
# 01/12/2025 07:42:47 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_quantized3/encoder_lora_epoch_2
# 01/12/2025 07:42:47 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_quantized3/decoder_lora_epoch_2

# quantized4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_quantized4 \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 4 \
    --trainable_nbit 16

# {   'eval/eval_ce_loss': 0.5827860837369891,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.8571428571428571,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.5827860837369891,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.8571428571428571,
#     'eval/train_valid_grid': 1.0}
# 01/12/2025 07:38:34 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_quantized4/eval_train_2.json
# 01/12/2025 07:38:34 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_quantized4/eval_eval_2.json
# 01/12/2025 07:38:38 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_quantized4/encoder_lora_epoch_2
# 01/12/2025 07:38:38 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_quantized4/decoder_lora_epoch_2

# combined
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_combined \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prefix_full \
    --no_lora \
    --tie_models \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16

# {   'eval/eval_ce_loss': 1.2427668211582517,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 1.2427668211582517,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/12/2025 07:11:26 - INFO - __main__ - Saved eval train generated text to ./encoder_decoder/outputs/test_combined/eval_train_4.json
# 01/12/2025 07:11:26 - INFO - __main__ - Saved eval eval generated text to ./encoder_decoder/outputs/test_combined/eval_eval_4.json
# 01/12/2025 07:11:29 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_combined/encoder_lora_epoch_4
# 01/12/2025 07:11:30 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_combined/conditioning_projection_epoch_4.pt

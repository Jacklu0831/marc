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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method prefix2prefix

# {   'eval/eval_ce_loss': 0.8129629574416738,
#     'eval/eval_competition_all_acc': 0.25,
#     'eval/eval_competition_sub_acc': 0.5714285714285714,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 0.8129629574416738,
#     'eval/train_competition_all_acc': 0.25,
#     'eval/train_competition_sub_acc': 0.5714285714285714,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/14/2025 19:27:21 - INFO - __main__ - Saved eval train pred gt to ./encoder_decoder/outputs/test_prefix2prefix/eval_train_2_pred_gt.json
# 01/14/2025 19:27:21 - INFO - __main__ - Saved eval eval pred gt to ./encoder_decoder/outputs/test_prefix2prefix/eval_eval_2_pred_gt.json
# 01/14/2025 19:27:21 - INFO - __main__ - Saved eval train vote to ./encoder_decoder/outputs/test_prefix2prefix/eval_train_2_vote.json
# 01/14/2025 19:27:21 - INFO - __main__ - Saved eval eval vote to ./encoder_decoder/outputs/test_prefix2prefix/eval_eval_2_vote.json
# 01/14/2025 19:27:24 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_prefix2prefix/encoder_lora_epoch_2
# 01/14/2025 19:27:28 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_prefix2prefix/decoder_lora_epoch_2

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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prefix_full

# {   'eval/eval_ce_loss': 0.9467761235843811,
#     'eval/eval_competition_all_acc': 0.25,
#     'eval/eval_competition_sub_acc': 0.5714285714285714,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 0.9467761235843811,
#     'eval/train_competition_all_acc': 0.25,
#     'eval/train_competition_sub_acc': 0.5714285714285714,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/14/2025 19:29:18 - INFO - __main__ - Saved eval train pred gt to ./encoder_decoder/outputs/test_hidden2prefix_full/eval_train_2_pred_gt.json
# 01/14/2025 19:29:18 - INFO - __main__ - Saved eval eval pred gt to ./encoder_decoder/outputs/test_hidden2prefix_full/eval_eval_2_pred_gt.json
# 01/14/2025 19:29:18 - INFO - __main__ - Saved eval train vote to ./encoder_decoder/outputs/test_hidden2prefix_full/eval_train_2_vote.json
# 01/14/2025 19:29:18 - INFO - __main__ - Saved eval eval vote to ./encoder_decoder/outputs/test_hidden2prefix_full/eval_eval_2_vote.json
# 01/14/2025 19:29:22 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prefix_full/encoder_lora_epoch_2
# 01/14/2025 19:29:26 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prefix_full/decoder_lora_epoch_2

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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt

# {   'eval/eval_ce_loss': 0.6032716794205564,
#     'eval/eval_competition_all_acc': 0.25,
#     'eval/eval_competition_sub_acc': 0.5714285714285714,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.888888888888889,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.6032716794205564,
#     'eval/train_competition_all_acc': 0.25,
#     'eval/train_competition_sub_acc': 0.5714285714285714,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.888888888888889,
#     'eval/train_valid_grid': 1.0}
# 01/14/2025 19:32:21 - INFO - __main__ - Saved eval train pred gt to ./encoder_decoder/outputs/test_hidden2prompt/eval_train_2_pred_gt.json
# 01/14/2025 19:32:21 - INFO - __main__ - Saved eval eval pred gt to ./encoder_decoder/outputs/test_hidden2prompt/eval_eval_2_pred_gt.json
# 01/14/2025 19:32:21 - INFO - __main__ - Saved eval train vote to ./encoder_decoder/outputs/test_hidden2prompt/eval_train_2_vote.json
# 01/14/2025 19:32:21 - INFO - __main__ - Saved eval eval vote to ./encoder_decoder/outputs/test_hidden2prompt/eval_eval_2_vote.json
# 01/14/2025 19:32:25 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prompt/encoder_lora_epoch_2
# 01/14/2025 19:32:28 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prompt/decoder_lora_epoch_2

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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt_full

# {   'eval/eval_ce_loss': 1.3648060049329485,
#     'eval/eval_competition_all_acc': 0.0,
#     'eval/eval_competition_sub_acc': 0.14285714285714285,
#     'eval/eval_correct_grid_dim': 0.5714285714285714,
#     'eval/eval_exact_acc': 0.14285714285714285,
#     'eval/eval_token_acc': 0.2698412698412698,
#     'eval/eval_valid_grid': 0.5714285714285714,
#     'eval/train_ce_loss': 1.3648060049329485,
#     'eval/train_competition_all_acc': 0.0,
#     'eval/train_competition_sub_acc': 0.14285714285714285,
#     'eval/train_correct_grid_dim': 0.5714285714285714,
#     'eval/train_exact_acc': 0.14285714285714285,
#     'eval/train_token_acc': 0.2698412698412698,
#     'eval/train_valid_grid': 0.5714285714285714}
# 01/14/2025 19:32:38 - INFO - __main__ - Saved eval train pred gt to ./encoder_decoder/outputs/test_hidden2prompt_full/eval_train_1_pred_gt.json
# 01/14/2025 19:32:38 - INFO - __main__ - Saved eval eval pred gt to ./encoder_decoder/outputs/test_hidden2prompt_full/eval_eval_1_pred_gt.json
# 01/14/2025 19:32:38 - INFO - __main__ - Saved eval train vote to ./encoder_decoder/outputs/test_hidden2prompt_full/eval_train_1_vote.json
# 01/14/2025 19:32:38 - INFO - __main__ - Saved eval eval vote to ./encoder_decoder/outputs/test_hidden2prompt_full/eval_eval_1_vote.json
# 01/14/2025 19:32:41 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prompt_full/encoder_lora_epoch_1
# 01/14/2025 19:32:45 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prompt_full/decoder_lora_epoch_1
# 01/14/2025 19:32:45 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_hidden2prompt_full/conditioning_projection_epoch_1.pt

# hidden2prompt_full_identity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_hidden2prompt_full_identity \
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
    --conditioning_method hidden2prompt_full_identity

# {   'eval/eval_ce_loss': 0.667765316457787,
#     'eval/eval_competition_all_acc': 0.25,
#     'eval/eval_competition_sub_acc': 0.5714285714285714,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 0.667765316457787,
#     'eval/train_competition_all_acc': 0.25,
#     'eval/train_competition_sub_acc': 0.5714285714285714,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/14/2025 19:37:55 - INFO - __main__ - Saved eval train pred gt to ./encoder_decoder/outputs/test_hidden2prompt_full_identity/eval_train_4_pred_gt.json
# 01/14/2025 19:37:55 - INFO - __main__ - Saved eval eval pred gt to ./encoder_decoder/outputs/test_hidden2prompt_full_identity/eval_eval_4_pred_gt.json
# 01/14/2025 19:37:55 - INFO - __main__ - Saved eval train vote to ./encoder_decoder/outputs/test_hidden2prompt_full_identity/eval_train_4_vote.json
# 01/14/2025 19:37:55 - INFO - __main__ - Saved eval eval vote to ./encoder_decoder/outputs/test_hidden2prompt_full_identity/eval_eval_4_vote.json
# 01/14/2025 19:37:58 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_hidden2prompt_full_identity/encoder_lora_epoch_4
# 01/14/2025 19:38:02 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_hidden2prompt_full_identity/decoder_lora_epoch_4
# 01/14/2025 19:38:02 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_hidden2prompt_full_identity/conditioning_projection_epoch_4.pt

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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --no_lora

# {   'eval/eval_ce_loss': 0.7249597536865622,
#     'eval/eval_competition_all_acc': 0.25,
#     'eval/eval_competition_sub_acc': 0.5714285714285714,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 0.7249597536865622,
#     'eval/train_competition_all_acc': 0.25,
#     'eval/train_competition_sub_acc': 0.5714285714285714,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/14/2025 19:43:24 - INFO - __main__ - Saved eval train pred gt to ./encoder_decoder/outputs/test_nolora/eval_train_9_pred_gt.json
# 01/14/2025 19:43:24 - INFO - __main__ - Saved eval eval pred gt to ./encoder_decoder/outputs/test_nolora/eval_eval_9_pred_gt.json
# 01/14/2025 19:43:24 - INFO - __main__ - Saved eval train vote to ./encoder_decoder/outputs/test_nolora/eval_train_9_vote.json
# 01/14/2025 19:43:24 - INFO - __main__ - Saved eval eval vote to ./encoder_decoder/outputs/test_nolora/eval_eval_9_vote.json
# 01/14/2025 19:43:32 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_nolora/encoder_lora_epoch_9
# 01/14/2025 19:43:41 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_nolora/decoder_lora_epoch_9

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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --tie_models

# {   'eval/eval_ce_loss': 0.5508727789669398,
#     'eval/eval_competition_all_acc': 0.25,
#     'eval/eval_competition_sub_acc': 0.5714285714285714,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.888888888888889,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.5508727789669398,
#     'eval/train_competition_all_acc': 0.25,
#     'eval/train_competition_sub_acc': 0.5714285714285714,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.888888888888889,
#     'eval/train_valid_grid': 1.0}
# 01/14/2025 19:41:20 - INFO - __main__ - Saved eval train pred gt to ./encoder_decoder/outputs/test_tiemodels/eval_train_2_pred_gt.json
# 01/14/2025 19:41:20 - INFO - __main__ - Saved eval eval pred gt to ./encoder_decoder/outputs/test_tiemodels/eval_eval_2_pred_gt.json
# 01/14/2025 19:41:20 - INFO - __main__ - Saved eval train vote to ./encoder_decoder/outputs/test_tiemodels/eval_train_2_vote.json
# 01/14/2025 19:41:20 - INFO - __main__ - Saved eval eval vote to ./encoder_decoder/outputs/test_tiemodels/eval_eval_2_vote.json
# 01/14/2025 19:41:24 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_tiemodels/encoder_lora_epoch_2

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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt_full \
    --encoder_name llama3b \
    --decoder_name llama1b

# 01/14/2025 19:45:49 - INFO - __main__ - Evaluation results:██████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.34it/s]
# {   'eval/eval_ce_loss': 0.7116662807883196,
#     'eval/eval_competition_all_acc': 0.25,
#     'eval/eval_competition_sub_acc': 0.5714285714285714,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.888888888888889,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.7116662807883196,
#     'eval/train_competition_all_acc': 0.25,
#     'eval/train_competition_sub_acc': 0.5714285714285714,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.888888888888889,
#     'eval/train_valid_grid': 1.0}
# 01/14/2025 19:45:49 - INFO - __main__ - Saved eval train pred gt to ./encoder_decoder/outputs/test_3bto1b/eval_train_2_pred_gt.json
# 01/14/2025 19:45:49 - INFO - __main__ - Saved eval eval pred gt to ./encoder_decoder/outputs/test_3bto1b/eval_eval_2_pred_gt.json
# 01/14/2025 19:45:49 - INFO - __main__ - Saved eval train vote to ./encoder_decoder/outputs/test_3bto1b/eval_train_2_vote.json
# 01/14/2025 19:45:49 - INFO - __main__ - Saved eval eval vote to ./encoder_decoder/outputs/test_3bto1b/eval_eval_2_vote.json
# 01/14/2025 19:45:55 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_3bto1b/encoder_lora_epoch_2
# 01/14/2025 19:45:58 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_3bto1b/decoder_lora_epoch_2
# 01/14/2025 19:45:59 - INFO - __main__ - Saved conditioning projection to ./encoder_decoder/outputs/test_3bto1b/conditioning_projection_epoch_2.pt

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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16

# {   'eval/eval_ce_loss': 0.7171367744782141,
#     'eval/eval_competition_all_acc': 0.25,
#     'eval/eval_competition_sub_acc': 0.5714285714285714,
#     'eval/eval_correct_grid_dim': 0.8571428571428571,
#     'eval/eval_exact_acc': 0.5714285714285714,
#     'eval/eval_token_acc': 0.7619047619047619,
#     'eval/eval_valid_grid': 0.8571428571428571,
#     'eval/train_ce_loss': 0.7171367744782141,
#     'eval/train_competition_all_acc': 0.25,
#     'eval/train_competition_sub_acc': 0.5714285714285714,
#     'eval/train_correct_grid_dim': 0.8571428571428571,
#     'eval/train_exact_acc': 0.5714285714285714,
#     'eval/train_token_acc': 0.7619047619047619,
#     'eval/train_valid_grid': 0.8571428571428571}
# 01/14/2025 19:57:16 - INFO - __main__ - Saved eval train pred gt to ./encoder_decoder/outputs/test_quantized/eval_train_9_pred_gt.json
# 01/14/2025 19:57:16 - INFO - __main__ - Saved eval eval pred gt to ./encoder_decoder/outputs/test_quantized/eval_eval_9_pred_gt.json
# 01/14/2025 19:57:16 - INFO - __main__ - Saved eval train vote to ./encoder_decoder/outputs/test_quantized/eval_train_9_vote.json
# 01/14/2025 19:57:16 - INFO - __main__ - Saved eval eval vote to ./encoder_decoder/outputs/test_quantized/eval_eval_9_vote.json
# 01/14/2025 19:57:18 - INFO - __main__ - Saved encoder to ./encoder_decoder/outputs/test_quantized/encoder_lora_epoch_9
# 01/14/2025 19:57:20 - INFO - __main__ - Saved decoder to ./encoder_decoder/outputs/test_quantized/decoder_lora_epoch_9

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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 4 \
    --trainable_nbit 32

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

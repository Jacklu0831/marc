# prefix2prefix
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_prefix2prefix \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method prefix2prefix \
    --weight_epoch 2 \
    --decoder_ce_loss

# {   'eval/ce_loss': 1.32301207951137,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.2857142857142857,
#     'eval/exact_acc': 0.0,
#     'eval/token_acc': 0.07936507936507937,
#     'eval/valid_grid': 0.2857142857142857}

# hidden2prefix_full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_hidden2prefix_full \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prefix_full \
    --weight_epoch 2 \
    --decoder_ce_loss

# {   'eval/ce_loss': 0.9467761235843811,
#     'eval/competition_all_acc': 0.25,
#     'eval/competition_sub_acc': 0.5714285714285714,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}

# hidden2prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_hidden2prompt \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --weight_epoch 2 \
    --decoder_ce_loss

# {   'eval/ce_loss': 0.6032716794205564,
#     'eval/competition_all_acc': 0.25,
#     'eval/competition_sub_acc': 0.5714285714285714,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.888888888888889,
#     'eval/valid_grid': 1.0}

# hidden2prompt_full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_hidden2prompt_full \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt_full \
    --weight_epoch 1 \
    --decoder_ce_loss

# {   'eval/ce_loss': 1.3648060049329485,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.14285714285714285,
#     'eval/correct_grid_dim': 0.5714285714285714,
#     'eval/exact_acc': 0.14285714285714285,
#     'eval/token_acc': 0.2698412698412698,
#     'eval/valid_grid': 0.5714285714285714}

# hidden2prompt_full_identity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_hidden2prompt_full_identity \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt_full_identity \
    --weight_epoch 4 \
    --decoder_ce_loss

# {   'eval/ce_loss': 0.667765316457787,
#     'eval/competition_all_acc': 0.25,
#     'eval/competition_sub_acc': 0.5714285714285714,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}

# no lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_nolora \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --no_lora \
    --weight_epoch 1 \
    --decoder_ce_loss

# {   'eval/ce_loss': 1.9298827988760812,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.0,
#     'eval/exact_acc': 0.0,
#     'eval/token_acc': 0.0,
#     'eval/valid_grid': 0.0}

# tie models
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_tiemodels \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --tie_models \
    --weight_epoch 2 \
    --decoder_ce_loss

# {   'eval/ce_loss': 0.5508727789669398,
#     'eval/competition_all_acc': 0.25,
#     'eval/competition_sub_acc': 0.5714285714285714,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.888888888888889,
#     'eval/valid_grid': 1.0}

# 3bto1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_3bto1b \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt_full \
    --encoder_name llama3b \
    --decoder_name llama1b \
    --weight_epoch 2 \
    --decoder_ce_loss

# {   'eval/ce_loss': 0.7116662807883196,
    # 'eval/competition_all_acc': 0.25,
    # 'eval/competition_sub_acc': 0.5714285714285714,
    # 'eval/correct_grid_dim': 1.0,
    # 'eval/exact_acc': 0.5714285714285714,
    # 'eval/token_acc': 0.888888888888889,
    # 'eval/valid_grid': 1.0}

# quantized
# GPU VRAM: 5742MiB 5742MiB
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_quantized \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --weight_epoch 9 \
    --decoder_ce_loss

# {   'eval/ce_loss': 0.7171367744782141,
#     'eval/competition_all_acc': 0.25,
#     'eval/competition_sub_acc': 0.5714285714285714,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}

# quantized2
# GPU VRAM: 6036MiB 6036MiB
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_quantized2 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 4 \
    --trainable_nbit 32 \
    --weight_epoch 10 \
    --decoder_ce_loss

# quantized3 (trained with >3 times lr because is late)
# GPU VRAM: 5950MiB 5950MiB
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_quantized3 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 3.6 \
    --trainable_nbit 32 \
    --weight_epoch 2 \
    --decoder_ce_loss

# quantized4 (trained with >3 times lr because is late)
# GPU VRAM: 5828MiB 5828MiB
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_quantized4 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 4 \
    --trainable_nbit 16 \
    --weight_epoch 2 \
    --decoder_ce_loss

# combined (trained with >3 times lr)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_combined \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --batch_size 2 \
    --conditioning_method hidden2prefix_full \
    --no_lora \
    --tie_models \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --weight_epoch 4 \
    --decoder_ce_loss
# prefix2prefix
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_prefix2prefix \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method prefix2prefix \
    --epoch 2

# {   'eval/ce_loss': 0.6310377611246493,
#     'eval/correct_grid_dim': 0.7142857142857143,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.6666666666666667,
#     'eval/valid_grid': 0.7142857142857143}

# hidden2prefix_shared
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_hidden2prefix_shared \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prefix_shared \
    --epoch 2

# {   'eval/ce_loss': 1.3218018407933414,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}

# hidden2prefix_full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_hidden2prefix_full \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prefix_full \
    --epoch 2

# {   'eval/ce_loss': 1.2037375001569413,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}

# hidden2prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_hidden2prompt \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --epoch 2

# {   'eval/ce_loss': 0.2515580686116924,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.8253968253968255,
#     'eval/valid_grid': 1.0}

# hidden2prompt_shared
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_hidden2prompt_shared \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt_shared \
    --epoch 7

# {   'eval/ce_loss': 1.0618006943592004,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}

# hidden2prompt_full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_hidden2prompt_full \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt_full \
    --epoch 2

# {   'eval/ce_loss': 0.338758490771787,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.888888888888889,
#     'eval/valid_grid': 1.0}

# no lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_nolora \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --no_lora \
    --epoch 2

# {   'eval/ce_loss': 0.30878038152669823,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.888888888888889,
#     'eval/valid_grid': 1.0}

# tie models
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_tiemodels \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --tie_models \
    --epoch 12

# {   'eval/ce_loss': 0.6156056267874581,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}

# 3bto1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_3bto1b \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt_full \
    --encoder_name llama3b \
    --decoder_name llama1b \
    --epoch 4

# {   'eval/ce_loss': 0.8358603226287025,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.888888888888889,
#     'eval/valid_grid': 1.0}

# quantized
# GPU VRAM: 5742MiB 5742MiB
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_quantized \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --epoch 2

# {   'eval/ce_loss': 0.8073542363043609,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.8730158730158731,
#     'eval/valid_grid': 1.0}

# quantized2
# GPU VRAM: 6036MiB 6036MiB
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_quantized2 \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 4 \
    --trainable_nbit 32 \
    --epoch 10

# {   'eval/ce_loss': 0.7623685662235532,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}

# quantized3 (trained with >3 times lr because is late)
# GPU VRAM: 5950MiB 5950MiB
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_quantized3 \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 3.6 \
    --trainable_nbit 32 \
    --epoch 2

# {   'eval/ce_loss': 0.5667667366755528,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.888888888888889,
#     'eval/valid_grid': 1.0}

# quantized4 (trained with >3 times lr because is late)
# GPU VRAM: 5828MiB 5828MiB
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_quantized4 \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prompt \
    --untrainable_nbit 4 \
    --trainable_nbit 16 \
    --epoch 2

# {   'eval/ce_loss': 0.5827860837369891,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.8571428571428571,
#     'eval/valid_grid': 1.0}

# combined (trained with >3 times lr)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --weight_dir test_combined \
    --eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit4/training \
    --batch_size 2 \
    --conditioning_method hidden2prefix_full \
    --no_lora \
    --tie_models \
    --untrainable_nbit 3.6 \
    --trainable_nbit 16 \
    --epoch 4

# {   'eval/ce_loss': 1.2427668211582517,
#     'eval/correct_grid_dim': 0.8571428571428571,
#     'eval/exact_acc': 0.5714285714285714,
#     'eval/token_acc': 0.7619047619047619,
#     'eval/valid_grid': 0.8571428571428571}
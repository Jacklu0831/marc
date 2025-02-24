# lmheads
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_lmhead \
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
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --conditioning_method prefix2prefix \
    --decoder_lm_head

# {   'eval/eval_ce_loss': 2.0270695005144392,
#     'eval/eval_competition_all_acc': 0.0,
#     'eval/eval_competition_sub_acc': 0.0,
#     'eval/eval_correct_grid_dim': 0.0,
#     'eval/eval_exact_acc': 0.0,
#     'eval/eval_token_acc': 0.0,
#     'eval/eval_valid_grid': 0.0,
#     'eval/train_ce_loss': 2.0270695005144392,
#     'eval/train_competition_all_acc': 0.0,
#     'eval/train_competition_sub_acc': 0.0,
#     'eval/train_correct_grid_dim': 0.0,
#     'eval/train_exact_acc': 0.0,
#     'eval/train_token_acc': 0.0,
#     'eval/train_valid_grid': 0.0}

# lmheads
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag test \
    --weight_dir test_lmhead \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
=======
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
>>>>>>> origin/main
    --conditioning_method prefix2prefix \
    --epoch 1

# {   'eval/ce_loss': 2.0270695005144392,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.0,
#     'eval/exact_acc': 0.0,
#     'eval/token_acc': 0.0,
#     'eval/valid_grid': 0.0}
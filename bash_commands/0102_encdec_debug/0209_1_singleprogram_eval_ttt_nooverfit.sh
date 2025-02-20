# try to save and load singleprogram model

# overfit1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag test_singleprogram_overfit \
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
    --min_prefix 4 \
    --max_prefix 4 \
    --num_epochs 100 \
    --samples_per_epoch 2 \
    --lr_embedding 0 \
    --lr_other 0 \
    --debug_fixed_order \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --num_workers 0 \
    --optimizer sgd \
    --log_every 1 \
    --eval_epoch 1 \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --tie_models

# {   'eval/eval_ce_loss': 4.652707576751709,
#     'eval/eval_competition_all_acc': 0.0,
#     'eval/eval_competition_sub_acc': 0.0,
#     'eval/eval_correct_grid_dim': 0.0,
#     'eval/eval_encoder_loss': 2.260035514831543,
#     'eval/eval_exact_acc': 0.0,
#     'eval/eval_kl_loss': 0.0,
#     'eval/eval_token_acc': 0.0,
#     'eval/eval_valid_grid': 0.0,
#     'eval/train_ce_loss': 4.652707576751709,
#     'eval/train_competition_all_acc': 0.0,
#     'eval/train_competition_sub_acc': 0.0,
#     'eval/train_correct_grid_dim': 0.0,
#     'eval/train_encoder_loss': 2.260035514831543,
#     'eval/train_exact_acc': 0.0,
#     'eval/train_kl_loss': 0.0,
#     'eval/train_token_acc': 0.0,
#     'eval/train_valid_grid': 0.0}

# evaluate the above
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/evaluate.py \
    --tag test \
    --weight_dir test_singleprogram_overfit \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit/training \
    --batch_size 2 \
    --tie_models

# {   'eval/ce_loss': 4.652707576751709,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.0,
#     'eval/encoder_loss': 2.260035514831543,
#     'eval/exact_acc': 0.0,
#     'eval/kl_loss': 0.0,
#     'eval/token_acc': 0.0,
#     'eval/ttt_provided': 0.0,
#     'eval/valid_grid': 0.0}

# now ttt (not even 1.5x the time of noprogram?)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/ttt.py \
    --tag test \
    --weight_dir test_singleprogram_overfit \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --max_samples_per_task 4 \
    --grad_accum_steps 1 \
    --optimizer sgd \
    --debug_no_aug \
    --max_grad_norm 1e8 \
    --lr 1e-2 \
    --num_epochs 100 \
    --log_every 1 \
    --save_epochs 100 \
    --tie_models

# ok now evaluate ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/evaluate.py \
    --tag test \
    --weight_dir test_singleprogram_overfit \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test_test_singleprogram_overfit \
    --ttt_weight_epoch 100 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --batch_size 2 \
    --tie_models

# {   'eval/ce_loss': 0.8127115368843079,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.5,
#     'eval/encoder_loss': 0.4492158703505993,
#     'eval/exact_acc': 0.0,
#     'eval/kl_loss': 0.0,
#     'eval/token_acc': 0.3888888888888889,
#     'eval/ttt_provided': 1.0,
#     'eval/valid_grid': 0.5}
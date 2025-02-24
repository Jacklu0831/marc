# try to save and load noprogram model

# overfit1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --tag test_noprogram \
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-4 \
    --lr_other 1e-4 \
    --debug_fixed_order \
    --max_grad_norm 1e8 \
    --num_workers 0 \
    --optimizer sgd \
    --min_num_pair 5 \
    --max_num_pair 5 \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --no_train_original \
    --log_every 1 \
    --grad_accum_steps 1

# {   'eval/eval_ce_loss': 0.16093355417251587,
#     'eval/eval_competition_all_acc': 0.0,
#     'eval/eval_competition_sub_acc': 0.5,
#     'eval/eval_correct_grid_dim': 1.0,
#     'eval/eval_exact_acc': 0.5,
#     'eval/eval_token_acc': 0.8333333333333333,
#     'eval/eval_valid_grid': 1.0,
#     'eval/train_ce_loss': 0.16093355417251587,
#     'eval/train_competition_all_acc': 0.0,
#     'eval/train_competition_sub_acc': 0.5,
#     'eval/train_correct_grid_dim': 1.0,
#     'eval/train_exact_acc': 0.5,
#     'eval/train_token_acc': 0.8333333333333333,
#     'eval/train_valid_grid': 1.0}

# evaluate the above
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag test \
    --weight_dir test_noprogram \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit/training \
    --batch_size 2

# {   'eval/ce_loss': 0.16093355417251587,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.5,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 0.5,
#     'eval/token_acc': 0.8333333333333333,
#     'eval/ttt_provided': 0.0,
#     'eval/valid_grid': 1.0}

# now ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --tag test \
    --weight_dir test_noprogram \
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
    --save_epochs 100

# ok now evaluate ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag test \
    --weight_dir test_noprogram \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test_test_noprogram \
    --ttt_weight_epoch 100 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --batch_size 2

# {   'eval/ce_loss': 5.038312792748911e-05,
#     'eval/competition_all_acc': 1.0,
#     'eval/competition_sub_acc': 1.0,
#     'eval/correct_grid_dim': 1.0,
#     'eval/exact_acc': 1.0,
#     'eval/token_acc': 1.0,
#     'eval/ttt_provided': 1.0,
#     'eval/valid_grid': 1.0}
# try to save and load autoregressive model

# overfit1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/train.py \
    --tag test_autoregressive_overfit \
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 50 \
    --lr_embedding 0 \
    --lr_program 0 \
    --lr_prior 0 \
    --lr_other 0 \
    --debug_fixed_order \
    --max_grad_norm 1e2 \
    --num_workers 0 \
    --optimizer sgd \
    --min_num_pair 5 \
    --max_num_pair 5 \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --no_train_original \
    --log_every 1 \
    --grad_accum_steps 1 \
    --train_no_sample \
    --eval_no_sample

# {   'eval/eval_ce_loss': 4.620311260223389,
#     'eval/eval_competition_all_acc': 0.0,
#     'eval/eval_competition_sub_acc': 0.0,
#     'eval/eval_correct_grid_dim': 0.0,
#     'eval/eval_exact_acc': 0.0,
#     'eval/eval_kl_loss': 427642.75,
#     'eval/eval_token_acc': 0.0,
#     'eval/eval_valid_grid': 0.0,
#     'eval/train_ce_loss': 4.620311260223389,
#     'eval/train_competition_all_acc': 0.0,
#     'eval/train_competition_sub_acc': 0.0,
#     'eval/train_correct_grid_dim': 0.0,
#     'eval/train_exact_acc': 0.0,
#     'eval/train_kl_loss': 427642.75,
#     'eval/train_token_acc': 0.0,
#     'eval/train_valid_grid': 0.0}

# evaluate the above
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/evaluate.py \
    --tag test \
    --weight_dir test_autoregressive_overfit \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit/training \
    --batch_size 2 \
    --no_sample

# {   'eval/ce_loss': 4.620311260223389,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.0,
#     'eval/exact_acc': 0.0,
#     'eval/kl_loss': 427642.75,
#     'eval/token_acc': 0.0,
#     'eval/ttt_provided': 0.0,
#     'eval/valid_grid': 0.0}

# now ttt (not even 2x the time of noprogram?)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/ttt.py \
    --tag test \
    --weight_dir test_autoregressive_overfit \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --max_samples_per_task 4 \
    --grad_accum_steps 1 \
    --optimizer sgd \
    --debug_no_aug \
    --max_grad_norm 1e2 \
    --lr 1e-3 \
    --num_epochs 100 \
    --log_every 1 \
    --save_epochs 100 \
    --no_sample

# ok now evaluate ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0210/evaluate.py \
    --tag test \
    --weight_dir test_autoregressive_overfit \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test_test_autoregressive_overfit \
    --ttt_weight_epoch 100 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --batch_size 2 \
    --no_sample

# {   'eval/ce_loss': 0.05306108482182026,
#     'eval/competition_all_acc': 0.5,
#     'eval/competition_sub_acc': 0.5,
#     'eval/correct_grid_dim': 0.5,
#     'eval/exact_acc': 0.5,
#     'eval/kl_loss': 138193.56640625,
#     'eval/token_acc': 0.5,
#     'eval/ttt_provided': 1.0,
#     'eval/valid_grid': 0.5}
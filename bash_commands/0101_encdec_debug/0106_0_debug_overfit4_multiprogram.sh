# repro
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1024_debug1/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_overfit \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --debug_random_pad \
    --conditioning_method hidden2prompt

# overfit1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_multiprogram_0203/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit_perfect/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit_perfect/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_program 1e-3 \
    --lr_other 1e-2 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --num_workers 0 \
    --optimizer sgd \
    --debug_random_pad \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --max_num_sample_program 4 \
    --min_num_pair_for_program 4 \
    --max_num_train_program 1 \
    --no_train_original \
    --debug_fixed_order \
    --limit_eval_to_max_program \
    --colon_encoding

# overfit1 multiprogram
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1025_bugged_multiprogram/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --min_chosen 5 \
    --max_chosen 5 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_program 1e-3 \
    --lr_other 1e-2 \
    --debug_overfit \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --num_workers 0 \
    --optimizer sgd \
    --debug_random_pad \
    --no_basic_aug \
    --max_num_sample_program 6 \
    --min_num_pair_for_program 4 \
    --max_num_train_program 2 \
    --no_train_original

# overfit4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1025_bugged_multiprogram/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_chosen 5 \
    --max_chosen 5 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_program 3e-4 \
    --lr_other 3e-3 \
    --debug_overfit \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --num_workers 0 \
    --optimizer sgd \
    --debug_random_pad \
    --no_basic_aug \
    --min_num_pair_for_program 4 \
    --max_num_program 1 \
    --no_train_original \
    --log_every 1

# aim for 30 secs for 100 iters, 8GB -> good
# why is encoder loss so high?
# test how fast actual training is, not limited to single program, look at debug_train_data
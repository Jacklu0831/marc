# overfit1 novae (weird, when debug_random_pad, need max_grad_norm to lower to fit properly and not nan)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-2 \
    --lr_program 1e-2 \
    --lr_other 1e-1 \
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
    --eval_no_sample \
    --debug_random_pad

# overfit1 vae (tried different kl_loss_lambda, does not fit...)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-2 \
    --lr_program 1e-2 \
    --lr_other 1e-1 \
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
    --eval_no_sample \
    --kl_loss_lambda 0.001 \
    --debug_random_pad

# overfit4 novae (harder to fit, but gradaccum and maxgradnorm help)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-3 \
    --lr_program 3e-3 \
    --lr_other 3e-2 \
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
    --grad_accum_steps 4 \
    --train_no_sample \
    --eval_no_sample \
    --debug_random_pad

# test if it runs on multigpu
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-3 \
    --lr_program 3e-3 \
    --lr_other 3e-2 \
    --debug_fixed_order \
    --max_grad_norm 1e2 \
    --num_workers 0 \
    --optimizer sgd \
    --min_num_pair 3 \
    --max_num_pair 5 \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --no_train_original \
    --log_every 1 \
    --grad_accum_steps 1 \
    --train_no_sample \
    --eval_no_sample

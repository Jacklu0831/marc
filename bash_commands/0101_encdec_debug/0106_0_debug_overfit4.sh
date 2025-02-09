# debug overfit 1 token (require SGD)
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
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --ntokens 2 \
    --max_grad_norm 1e8 \
    --num_workers 0 \
    --optimizer sgd \
    --debug_random_pad \
    --no_basic_aug

# debug overfit 1 token (require SGD)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1024_debug1/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --ntokens 2 \
    --max_grad_norm 1e8 \
    --num_workers 0 \
    --optimizer sgd \
    --debug_random_pad \
    --conditioning_method hidden2prompt

# test gradient search prefix2prefix
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder5/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_ttt/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_ttt/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 8 \
    --eval_batch_size 1 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_optimizer sgd \
    --gs_max_grad_norm 1e8 \
    --conditioning_method prefix2prefix \
    --gs_iters 1000

# test gradient search hidden2prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder5/train.py \
    --tag test \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_ttt/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit2_ttt/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 8 \
    --eval_batch_size 1 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_optimizer sgd \
    --gs_max_grad_norm 1e8 \
    --gs_iters 1000 \
    --ntokens 64 # 8 times of prefix2prefix to match dimension

# hidden2prompt <<< prefix2prefix, even with same dimension
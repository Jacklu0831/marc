# torch.Size([2, 569])
# torch.Size([2, 143])

# debug overfit 1 token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder2/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --debug_random_pad \
    --conditioning_method hidden2prompt \
    --num_workers 0

# debug overfit 1 token debug extra train
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder5/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit_extra_train/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit_extra_train/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --debug_random_pad \
    --conditioning_method hidden2prompt \
    --num_workers 0 \
    --extra_train_ratio 1.0 \
    --debug_extra_no_aug

# debug overfit 1 token (require SGD) full dataset
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder2/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --num_workers 0 \
    --augment_ratio 0.3 \
    --num_pair_sigma 2.0 \
    --extra_train_ratio 1.0

# debug ddp error
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --conditioning_method hidden2prompt_full \
    --tie_models \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --max_grad_norm 1e8 \
    --optimizer sgd

# debug overfit 2 token (did not work)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test2 \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 1000 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 2

# debug overfit 4 token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test4 \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 1000 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 4


# debug multiepoch and stuff
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test1 \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 2 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1 \
    --wandb

# idk
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug/evaluation \
=======
    --train_data_dir ./data/re-arc/train_data_debug/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug/evaluation \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 2 \
    --samples_per_epoch 160 \
    --wandb

# 5 and 3 eval samples
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_sub/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_sub/evaluation \
=======
    --train_data_dir ./data/re-arc/train_data/tasks \
    --eval_train_dir ./data/re-arc/arc_original_sub/training \
    --eval_eval_dir ./data/re-arc/arc_original_sub/evaluation \
>>>>>>> origin/main
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 16 \
    --lr_embedding 0.0 \
    --lr_other 0.0

# debug overfit 1 token compact grids
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test1 \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1 \
    --compact_grids

# memory stress test (62.3GB)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test1 \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 4 \
    --num_virtual_tokens 1 \
    --dummy_seq_enc_len 8192 \
    --dummy_seq_dec_len 3613 \
    --flash_attn

# debug overfit 1 token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1

# debug overfit 1 token lmheads flashattn
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag test \
<<<<<<< HEAD
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit/training \
=======
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
>>>>>>> origin/main
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 1 \
    --samples_per_epoch 500 \
    --lr_embedding 3e-4 \
    --lr_other 3e-3 \
    --debug_fixed_train_order \
    --invar_loss_lambda 0.0 \
    --grad_accum_steps 1 \
    --no_gradient_checkpointing \
    --num_virtual_tokens 1 \
    --encoder_lm_head \
    --decoder_lm_head \
    --flash_attn
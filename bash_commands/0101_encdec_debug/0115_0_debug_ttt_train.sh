# quick ckpt 1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/train.py \
    --tag test_ttt_1 \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --tie_models \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --num_epochs 1 \
    --samples_per_epoch 2 \
    --grad_accum_steps 1 \
    --conditioning_method hidden2prompt_full

# quick ckpt 2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/train.py \
    --tag test_ttt_2 \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --num_epochs 1 \
    --samples_per_epoch 2 \
    --grad_accum_steps 1 \
    --conditioning_method prefix2prefix

# train1 ttt partiallora single gpu
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/ttt.py \
    --tag test1_partiallora_single_gpu \
    --weight_dir test_ttt_1 \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --tie_models \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method hidden2prompt_full \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 4 \
    --grad_accum_steps 1 \
    --optimizer sgd \
    --debug_no_aug \
    --max_grad_norm 1e8 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --num_epochs 100 \
    --log_every 1 \
    --save_epochs 100

# train1 ttt partiallora multi gpu
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/ttt.py \
    --tag test1_partiallora_multi_gpu \
    --weight_dir test_ttt_1 \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --tie_models \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method hidden2prompt_full \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 4 \
    --grad_accum_steps 1 \
    --optimizer sgd \
    --debug_no_aug \
    --max_grad_norm 1e8 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --num_epochs 100 \
    --log_every 1 \
    --save_epochs 100

# train2 ttt partiallora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/ttt.py \
    --tag test2_partiallora \
    --weight_dir test_ttt_2 \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4_ttt/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method prefix2prefix \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 4 \
    --grad_accum_steps 1 \
    --optimizer sgd \
    --debug_no_aug \
    --max_grad_norm 1e8 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --num_epochs 100 \
    --debug_no_aug \
    --log_every 1 \
    --save_epochs 100

# train1 ttt fulllora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/ttt.py \
    --tag test1_fulllora \
    --weight_dir test_ttt_1 \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4_ttt/training \
    --tie_models \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method hidden2prompt_full \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 4 \
    --grad_accum_steps 1 \
    --optimizer sgd \
    --debug_no_aug \
    --max_grad_norm 1e8 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --num_epochs 100 \
    --debug_no_aug \
    --log_every 1 \
    --save_epochs 100 \
    --full_lora

# train2 ttt fulllora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/ttt.py \
    --tag test2_fulllora \
    --weight_dir test_ttt_2 \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4_ttt/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method prefix2prefix \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 4 \
    --grad_accum_steps 1 \
    --optimizer sgd \
    --debug_no_aug \
    --max_grad_norm 1e8 \
    --lr_embedding 1e-3 \
    --lr_other 1e-2 \
    --num_epochs 100 \
    --debug_no_aug \
    --log_every 1 \
    --save_epochs 100 \
    --full_lora
# nemo8b
accelerate launch --main_process_port $MASTER_PORT --num_processes 1 --mixed_precision bf16 encoder_decoder2/train.py \
    --tag test_nemo8b \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 50 \
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
    --encoder_pad_side left \
    --decoder_pad_side left \
    --decoder_gen_pad_side left \
    --conditioning_method hidden2prompt \
    --encoder_name nemo8b \
    --decoder_name nemo8b \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --untrainable_nbit 4 \
    --samples_per_epoch 8

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder2/evaluate.py \
    --tag test \
    --weight_dir test_nemo8b \
    --weight_epoch 1 \
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --ntokens 2 \
    --encoder_name nemo8b \
    --decoder_name nemo8b \
    --encoder_pad_side left \
    --decoder_pad_side left \
    --decoder_gen_pad_side left \
    --untrainable_nbit 4 \
    --decoder_ce_loss

# llama3b
accelerate launch --main_process_port $MASTER_PORT --num_processes 1 --mixed_precision bf16 encoder_decoder2/train.py \
    --tag test_llama3b \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 50 \
    --samples_per_epoch 50 \
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
    --encoder_pad_side right \
    --decoder_pad_side right \
    --decoder_gen_pad_side left \
    --conditioning_method hidden2prompt \
    --encoder_name llama3b \
    --decoder_name llama3b \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --untrainable_nbit 4

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder2/evaluate.py \
    --tag test \
    --weight_dir test_llama3b \
    --weight_epoch 1 \
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --ntokens 2 \
    --encoder_name llama3b \
    --decoder_name llama3b \
    --untrainable_nbit 4 \
    --decoder_ce_loss

# llama8b
accelerate launch --main_process_port $MASTER_PORT --num_processes 1 --mixed_precision bf16 encoder_decoder/train.py \
    --tag test_llama8b \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
    --num_epochs 50 \
    --samples_per_epoch 50 \
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
    --encoder_pad_side right \
    --decoder_pad_side right \
    --decoder_gen_pad_side left \
    --conditioning_method hidden2prompt \
    --encoder_name llama8b \
    --decoder_name llama8b \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --dummy_enc_len 8192 \
    --untrainable_nbit 4

# slight deviation, maybe just because big model?
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder2/evaluate.py \
    --tag test \
    --weight_dir test_llama8b \
    --weight_epoch 1 \
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --ntokens 2 \
    --encoder_name llama8b \
    --decoder_name llama8b \
    --untrainable_nbit 4 \
    --decoder_ce_loss

# llama1b
accelerate launch --main_process_port $MASTER_PORT --num_processes 1 --mixed_precision bf16 encoder_decoder2/train.py \
    --tag test_llama1b \
    --train_data_dir /scratch/zy3101/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --min_prefix 4 \
    --max_prefix 4 \
    --augment_ratio 0.0 \
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
    --encoder_pad_side right \
    --decoder_pad_side right \
    --decoder_gen_pad_side left \
    --conditioning_method hidden2prompt

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder2/evaluate.py \
    --tag test \
    --weight_dir test_llama1b \
    --weight_epoch 1 \
    --data_dir /scratch/zy3101/re-arc/arc_original_debug_overfit4/training \
    --ntokens 2 \
    --decoder_ce_loss

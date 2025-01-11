# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0110_multigpu/0111_4_combine.sh

# combine noprojectkv invar0.001
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0111_combine_noprojectkv_invar0.001 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.3 \
    --invar_loss_lambda 0.001 \
    --compact_grids \
    --max_seq_len 5120 \
    --max_grad_norm 10.0 \
    --flash_attn \
    --decoder_lm_head \
    --wandb

# combine projectkv invar0.001
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0111_combine_projectkv_invar0.001 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.3 \
    --invar_loss_lambda 0.001 \
    --compact_grids \
    --max_seq_len 5120 \
    --max_grad_norm 10.0 \
    --flash_attn \
    --decoder_lm_head \
    --project_kv \
    --wandb

# combine noprojectkv invar0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0111_combine_noprojectkv_invar0.01 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.3 \
    --invar_loss_lambda 0.01 \
    --compact_grids \
    --max_seq_len 5120 \
    --max_grad_norm 10.0 \
    --flash_attn \
    --decoder_lm_head \
    --wandb

# combine projectkv invar0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0111_combine_projectkv_invar0.01 \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.3 \
    --invar_loss_lambda 0.01 \
    --compact_grids \
    --max_seq_len 5120 \
    --max_grad_norm 10.0 \
    --flash_attn \
    --decoder_lm_head \
    --project_kv \
    --wandb

# Submitted batch job 55753732
# Submitted batch job 55753733
# Submitted batch job 55753734
# Submitted batch job 55753735
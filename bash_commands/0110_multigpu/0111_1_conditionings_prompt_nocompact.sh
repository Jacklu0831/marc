# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0110_multigpu/0111_1_conditionings_prompt_nocompact.sh

# conditionings prompt ckpting
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0111_conditionings_prompt_ckpting \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --wandb

# conditionings prompt ckpting nocompact
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0111_conditionings_prompt_nocompact \
    --train_data_dir /scratch/yl11330/re-arc/train_data/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --eval_epochs 1 \
    --num_epochs 20 \
    --samples_per_epoch 20000 \
    --augment_ratio 0.0 \
    --invar_loss_lambda 0.0 \
    --max_seq_len 8192 \
    --conditioning_method hidden2prompt \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --wandb

# cancelled
# Submitted batch job 55755462
# Submitted batch job 55755463
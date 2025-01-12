# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0110_multigpu/0112_2_conditionings_more_prompt.sh
# 0111_0_conditionings.sh compares 4 types of conditioning, this adds full/shared projection to hidden2prompt

# conditionings promptshared
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/train.py \
    --tag 0112_conditionings_promptshared \
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
    --conditioning_method hidden2prompt_shared \
    --wandb

# conditionings promptfull
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/train.py \
    --tag 0112_conditionings_promptfull \
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
    --conditioning_method hidden2prompt_full \
    --wandb

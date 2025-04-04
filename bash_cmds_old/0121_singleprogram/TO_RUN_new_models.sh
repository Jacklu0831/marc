# python make_sbatch.py --ngpu 2 --time 96 --bash_files bash_commands/0121/0121_1_new_models.sh
# python make_sbatch.py --ngpu 4 --time 96 --bash_files bash_commands/0121/0121_1_new_models.sh

# llama3b uncensored
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_llama3b_uncensored \
    --encoder_name llama3b_uncensored \
    --decoder_name llama3b_uncensored \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --untrainable_nbit 4 \
    --wandb

# nemo8b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_nemo8b \
    --encoder_name nemo8b \
    --decoder_name nemo8b \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --untrainable_nbit 4 \
    --encoder_pad_side left \
    --decoder_pad_side left \
    --wandb

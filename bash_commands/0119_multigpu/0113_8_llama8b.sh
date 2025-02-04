# python make_sbatch.py --ngpu 4 --time 96 --bash_files bash_commands/0119_multigpu/0113_8_llama8b.sh
# 4gpus

# llama8b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_llama8b \
    --encoder_gradient_checkpointing \
    --decoder_gradient_checkpointing \
    --encoder_name llama8b \
    --decoder_name llama8b \
    --untrainable_nbit 4 \
    --wandb

# Submitted batch job 56077352
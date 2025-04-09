# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121/0121_4_architectsaug.sh

# architectsaug
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_architectsaug \
    --augment_type architects \
    --augment_ratio 1.0 \
    --wandb

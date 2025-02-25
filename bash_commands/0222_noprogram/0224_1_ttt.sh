# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0222_noprogram/0224_1_ttt.sh
# time: <4 hrs on single a100
# disk memory: 293M for lora * 5 epoch * 80 task = 118GB

# ttt noprogram base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --select_tasks_path task_info_selected.csv \
    --weight_dir test \
    --weight_epoch 3 \
    --tag test \
    --num_epochs 5 \
    --save_epochs 1


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --select_tasks_path task_info_selected.csv \
    --weight_dir test \
    --weight_epoch 3 \
    --tag test \
    --num_epochs 5 \
    --save_epochs 1

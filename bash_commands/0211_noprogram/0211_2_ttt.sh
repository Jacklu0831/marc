# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0211_noprogram/0211_2_ttt.sh
# time: <4 hrs on single a100
# disk memory: 293M for lora * 5 epoch * 80 task = 118GB

# ttt noprogram base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --weight_dir 0211_noprogram_base \
    --weight_epoch 24 \
    --tag base \
    --num_epochs 5 \
    --save_epochs 1

# # TOREMOVE, just testing
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
#     --weight_dir test_noprogram \
#     --weight_epoch 1 \
#     --tag test \
#     --num_epochs 5 \
#     --save_epochs 1 \
#     --aug_type extra

# Submitted batch job 57353572
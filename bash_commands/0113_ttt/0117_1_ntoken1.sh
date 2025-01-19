# python make_sbatch.py --gb 64 --ngpu 1 --time 48 --bash_files bash_commands/0113_ttt/0117_1_ntoken1.sh
# 293MB per encoder.pt (compared to 145MB for ttt paper lora)
# 293MB * 5epoch * 3run * 80task = 351.6GB

# lr1e-4 flashattn
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/ttt.py \
    --tag lr1e-4_flashattn \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected.csv \
    --num_epochs 5 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --max_samples_per_task 250 \
    --lr_embedding 1e-5 \
    --lr_other 1e-4

# lr1e-4 noflashattn
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/ttt.py \
    --tag lr1e-4_noflashattn \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected.csv \
    --num_epochs 5 \
    --compact_grids \
    --max_seq_len 5120 \
    --tie_models \
    --num_virtual_tokens 1 \
    --max_samples_per_task 250 \
    --lr_embedding 1e-5 \
    --lr_other 1e-4

# lr5e-5 flashattn
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/ttt.py \
    --tag lr5e-5_flashattn \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected.csv \
    --num_epochs 5 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --max_samples_per_task 250 \
    --lr_embedding 5e-6 \
    --lr_other 5e-5

# Submitted batch job 55987466
# Submitted batch job 55987467
# Submitted batch job 55987468
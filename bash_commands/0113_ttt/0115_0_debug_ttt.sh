# python make_sbatch.py --gb 64 --ngpu 2 --time 48 --bash_files bash_commands/0113_ttt/_0113_1_debug_ttt.sh
# for ttt try partial/full lora, 16bit/32bit

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test_partiallora \
    --weight_dir test_base \
    --weight_epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 1 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full_identity \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 8




accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test \
    --weight_dir test_base \
    --weight_epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 1 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full_identity \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 8




accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/ttt.py \
    --tag test_fulllora \
    --weight_dir test_base \
    --weight_epoch 1 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected_test.csv \
    --eval_epochs 1 \
    --num_epochs 1 \
    --invar_loss_lambda 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --conditioning_method hidden2prompt_full_identity \
    --encoder_loss_lambda 1.0 \
    --max_samples_per_task 8 \
    --full_lora

# for eval 80 task
# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_commands/0317_noprogram/0322_0_eval_ttt.sh --rtx8000

# for eval 400 task
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0317_noprogram/0322_0_eval_ttt.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0317_noprogram/0322_0_eval_ttt.sh








# eval 0317_noprogram_base epoch0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --select_tasks_path task_info_selected.csv

# eval 0317_noprogram_base epoch0 alltask
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch0_alltask \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24







# ttt 0317_noprogram_base augextra
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --lr_scheduler constant \
    --no_bos \
    --select_tasks_path task_info_selected.csv \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --tag augextra \
    --num_epochs 10 \
    --save_epochs 2 \
    --aug_type extra

# ttt 0317_noprogram_base augd8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --lr_scheduler constant \
    --no_bos \
    --select_tasks_path task_info_selected.csv \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --tag augd8 \
    --num_epochs 10 \
    --save_epochs 2 \
    --aug_type d8







# eval 0317_noprogram_base augd8 epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_noprogram_base \
    --ttt_weight_epoch 2 \
    --select_tasks_path task_info_selected.csv

# eval 0317_noprogram_base augd8 epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_noprogram_base \
    --ttt_weight_epoch 4 \
    --select_tasks_path task_info_selected.csv

# eval 0317_noprogram_base augd8 epoch6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch6 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_noprogram_base \
    --ttt_weight_epoch 6 \
    --select_tasks_path task_info_selected.csv

# eval 0317_noprogram_base augd8 epoch8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_noprogram_base \
    --ttt_weight_epoch 8 \
    --select_tasks_path task_info_selected.csv

# eval 0317_noprogram_base augd8 epoch10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch10 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_noprogram_base \
    --ttt_weight_epoch 10 \
    --select_tasks_path task_info_selected.csv
























# eval 0317_noprogram_base augextra epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_noprogram_base \
    --ttt_weight_epoch 2 \
    --select_tasks_path task_info_selected.csv

# eval 0317_noprogram_base augextra epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_noprogram_base \
    --ttt_weight_epoch 4 \
    --select_tasks_path task_info_selected.csv

# eval 0317_noprogram_base augextra epoch6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch6 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_noprogram_base \
    --ttt_weight_epoch 6 \
    --select_tasks_path task_info_selected.csv

# eval 0317_noprogram_base augextra epoch8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_noprogram_base \
    --ttt_weight_epoch 8 \
    --select_tasks_path task_info_selected.csv

# eval 0317_noprogram_base augextra epoch10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag epoch10 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_noprogram_base \
    --ttt_weight_epoch 10 \
    --select_tasks_path task_info_selected.csv








# eval original
# Submitted batch job 58666095 # 0.1875
# Submitted batch job 58666096 # 0.13 (slightly higher than 12.75 from before, due to batch size difference and noflashattn)

# ttt
# Submitted batch job 58656286 # 5.4hr
# Submitted batch job 58656287 # 3.8hr

# eval ttt augd8
# Submitted batch job 58758401 # 0.2125
# Submitted batch job 58758402 # 0.2625
# Submitted batch job 58758403 # 0.2625
# Submitted batch job 58758404 # 0.2625
# Submitted batch job 58758405 # 0.275

# eval ttt augextra
# Submitted batch job 58758410 # 0.2125
# Submitted batch job 58758411 # 0.225
# Submitted batch job 58758412 # 0.2625
# Submitted batch job 58758413 # 0.2
# Submitted batch job 58758414 # 0.2125

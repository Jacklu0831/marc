# for eval, run rtx8000
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0317_autoregressive/0322_0_eval_ttt.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0317_autoregressive/0322_0_eval_ttt.sh








# eval 0317_arlongcache_base epoch0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch0 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --select_tasks_path task_info_selected.csv

# eval 0317_arlongcache_base epoch0 alltask
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch0_alltask \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24







# ttt 0317_arlongcache_base augextra
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/ttt.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --select_tasks_path task_info_selected.csv \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --tag augextra \
    --num_epochs 10 \
    --save_epochs 2 \
    --aug_type extra

# ttt 0317_arlongcache_base augd8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/ttt.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --select_tasks_path task_info_selected.csv \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --tag augd8 \
    --num_epochs 10 \
    --save_epochs 2 \
    --aug_type d8







# eval 0317_arlongcache_base augd8 epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch2 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_arlongcache_base \
    --ttt_weight_epoch 2 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv

# eval 0317_arlongcache_base augd8 epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch4 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_arlongcache_base \
    --ttt_weight_epoch 4 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv

# eval 0317_arlongcache_base augd8 epoch6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch6 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_arlongcache_base \
    --ttt_weight_epoch 6 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv

# eval 0317_arlongcache_base augd8 epoch8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch8 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_arlongcache_base \
    --ttt_weight_epoch 8 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv

# eval 0317_arlongcache_base augd8 epoch10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch10 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_arlongcache_base \
    --ttt_weight_epoch 10 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv
















# eval 0317_arlongcache_base augextra epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch2 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_arlongcache_base \
    --ttt_weight_epoch 2 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv

# eval 0317_arlongcache_base augextra epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch4 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_arlongcache_base \
    --ttt_weight_epoch 4 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv

# eval 0317_arlongcache_base augextra epoch6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch6 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_arlongcache_base \
    --ttt_weight_epoch 6 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv

# eval 0317_arlongcache_base augextra epoch8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch8 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_arlongcache_base \
    --ttt_weight_epoch 8 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv

# eval 0317_arlongcache_base augextra epoch10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --no_bos \
    --tag epoch10 \
    --weight_dir 0317_arlongcache_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_arlongcache_base \
    --ttt_weight_epoch 10 \
    --leave_ns 0 \
    --select_tasks_path task_info_selected.csv



# eval original
# Submitted batch job 58666093 # 0.20
# Submitted batch job 58666094 # 0.1425 (exactly as before)

# ttt
# Submitted batch job 58656291 # 14.8hr
# Submitted batch job 58656292 # 12.1hr

# eval ttt augd8
# Submitted batch job 58790480 # 0.25
# Submitted batch job 58790481 # 0.225
# Submitted batch job 58790482 # 0.25
# Submitted batch job 58790483 # 0.25
# Submitted batch job 58790484 # 0.25

# eval ttt augextra
# Submitted batch job 58790487 # 0.2125
# Submitted batch job 58790488 # 0.2
# Submitted batch job 58790489 # 0.2625
# Submitted batch job 58790490 # 0.275
# Submitted batch job 58790491 # 0.25
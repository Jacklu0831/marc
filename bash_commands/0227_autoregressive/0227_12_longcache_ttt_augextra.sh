# for eval, run rtx8000
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0227_autoregressive/0227_12_longcache_ttt_augextra.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0227_autoregressive/0227_12_longcache_ttt_augextra.sh






# # ttt 0227_arlongcache_demondropout0.1 augextra
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/ttt.py \
#     --lr_scheduler constant \
#     --token_weighted_loss \
#     --select_tasks_path task_info_selected.csv \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18 \
#     --tag augextra \
#     --num_epochs 5 \
#     --save_epochs 1 \
#     --aug_type d8



# eval 0227_arlongcache_demondropout0.1 augextra epoch1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --tag epoch1 \
    --weight_dir 0227_arlongcache_demondropout0.1 \
    --weight_epoch 18 \
    --ttt_weight_dir ttt_augextra_0227_arlongcache_demondropout0.1 \
    --ttt_weight_epoch 1 \
    --select_tasks_path task_info_selected.csv

# eval 0227_arlongcache_demondropout0.1 augextra epoch2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --tag epoch2 \
    --weight_dir 0227_arlongcache_demondropout0.1 \
    --weight_epoch 18 \
    --ttt_weight_dir ttt_augextra_0227_arlongcache_demondropout0.1 \
    --ttt_weight_epoch 2 \
    --select_tasks_path task_info_selected.csv

# eval 0227_arlongcache_demondropout0.1 augextra epoch3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --tag epoch3 \
    --weight_dir 0227_arlongcache_demondropout0.1 \
    --weight_epoch 18 \
    --ttt_weight_dir ttt_augextra_0227_arlongcache_demondropout0.1 \
    --ttt_weight_epoch 3 \
    --select_tasks_path task_info_selected.csv

# eval 0227_arlongcache_demondropout0.1 augextra epoch4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --tag epoch4 \
    --weight_dir 0227_arlongcache_demondropout0.1 \
    --weight_epoch 18 \
    --ttt_weight_dir ttt_augextra_0227_arlongcache_demondropout0.1 \
    --ttt_weight_epoch 4 \
    --select_tasks_path task_info_selected.csv

# eval 0227_arlongcache_demondropout0.1 augextra epoch5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --tag epoch5 \
    --weight_dir 0227_arlongcache_demondropout0.1 \
    --weight_epoch 18 \
    --ttt_weight_dir ttt_augextra_0227_arlongcache_demondropout0.1 \
    --ttt_weight_epoch 5 \
    --select_tasks_path task_info_selected.csv




# EVALUATE BEST TTT WITH VOTING (can try different config for leave_ns)
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18 \
#     --ttt_weight_dir ttt_augextra_0227_arlongcache_demondropout0.1 \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv
#     --select_tasks_path task_info_selected.csv
#     --leave_ns 1 \
#     --permute_n 2 \
#     --augment_n 5 \
#     --leave_ns_inc




# Submitted batch job 57918104 # subset of 80 tasks, seems like it's easier
# Submitted batch job 57918105 # eval on all task matches the previous training run!
# Submitted batch job 57918117 # ttt

# eval ttt on rtx8000
# Submitted batch job 58107150
# Submitted batch job 58107151
# Submitted batch job 58107152
# Submitted batch job 58107153
# Submitted batch job 58107154
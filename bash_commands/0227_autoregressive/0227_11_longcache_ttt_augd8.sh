# for eval, run rtx8000
# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0227_autoregressive/0227_11_longcache_ttt_augd8.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0227_autoregressive/0227_11_longcache_ttt_augd8.sh





# # eval 0227_arlongcache_demondropout0.1 epoch0
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/evaluate.py \
#     --tag epoch0 \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18 \
#     --select_tasks_path task_info_selected.csv

# # eval 0227_arlongcache_demondropout0.1 epoch0 alltask
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/evaluate.py \
#     --tag epoch0_alltask \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18





# ttt 0227_arlongcache_demondropout0.1 augd8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/ttt.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --select_tasks_path task_info_selected.csv \
    --weight_dir 0227_arlongcache_demondropout0.1 \
    --weight_epoch 18 \
    --tag augd8 \
    --num_epochs 5 \
    --save_epochs 1 \
    --aug_type d8



# # eval 0227_arlongcache_demondropout0.1 augd8 epoch1
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/evaluate.py \
#     --tag epoch1 \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18 \
#     --ttt_weight_dir ttt_augd8_0227_arlongcache_demondropout0.1 \
#     --ttt_weight_epoch 1 \
#     --select_tasks_path task_info_selected.csv

# # eval 0227_arlongcache_demondropout0.1 augd8 epoch2
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/evaluate.py \
#     --tag epoch2 \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18 \
#     --ttt_weight_dir ttt_augd8_0227_arlongcache_demondropout0.1 \
#     --ttt_weight_epoch 2 \
#     --select_tasks_path task_info_selected.csv

# # eval 0227_arlongcache_demondropout0.1 augd8 epoch3
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/evaluate.py \
#     --tag epoch3 \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18 \
#     --ttt_weight_dir ttt_augd8_0227_arlongcache_demondropout0.1 \
#     --ttt_weight_epoch 3 \
#     --select_tasks_path task_info_selected.csv

# # eval 0227_arlongcache_demondropout0.1 augd8 epoch4
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/evaluate.py \
#     --tag epoch4 \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18 \
#     --ttt_weight_dir ttt_augd8_0227_arlongcache_demondropout0.1 \
#     --ttt_weight_epoch 4 \
#     --select_tasks_path task_info_selected.csv

# # eval 0227_arlongcache_demondropout0.1 augd8 epoch5
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18 \
#     --ttt_weight_dir ttt_augd8_0227_arlongcache_demondropout0.1 \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv




# EVALUATE BEST TTT WITH VOTING (can try different config for leave_ns)
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0227_arlongcache_demondropout0.1 \
#     --weight_epoch 18 \
#     --ttt_weight_dir ttt_augd8_0227_arlongcache_demondropout0.1 \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv
#     --select_tasks_path task_info_selected.csv
#     --leave_ns 1 \
#     --permute_n 2 \
#     --augment_n 5 \
#     --leave_ns_inc




# Submitted batch job 57918082 # subset of 80 tasks, seems like it's easier
# Submitted batch job 57918083 # eval on all task matches the previous training run!
# Submitted batch job 57918086 # ttt
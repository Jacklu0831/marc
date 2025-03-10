# for eval, run rtx8000
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0227_autoregressive/0309_6_longcache_ttt_augextra.sh --rtx8000

# for ttt, run a100
# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0227_autoregressive/0309_6_longcache_ttt_augextra.sh



# ttt 0227_arlongcache_ntokens4 augextra
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/ttt.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --select_tasks_path task_info_selected.csv \
    --weight_dir 0227_arlongcache_ntokens4 \
    --weight_epoch 28 \
    --tag augextra \
    --num_epochs 10 \
    --save_epochs 2 \
    --aug_type extra



# # eval 0227_arlongcache_ntokens4 augextra epoch2
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
#     --tag epoch2 \
#     --weight_dir 0227_arlongcache_ntokens4 \
#     --weight_epoch 28 \
#     --ttt_weight_dir ttt_augextra_0227_arlongcache_ntokens4 \
#     --ttt_weight_epoch 2 \
#     --select_tasks_path task_info_selected.csv

# # eval 0227_arlongcache_ntokens4 augextra epoch4
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
#     --tag epoch4 \
#     --weight_dir 0227_arlongcache_ntokens4 \
#     --weight_epoch 28 \
#     --ttt_weight_dir ttt_augextra_0227_arlongcache_ntokens4 \
#     --ttt_weight_epoch 4 \
#     --select_tasks_path task_info_selected.csv

# # eval 0227_arlongcache_ntokens4 augextra epoch6
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
#     --tag epoch6 \
#     --weight_dir 0227_arlongcache_ntokens4 \
#     --weight_epoch 28 \
#     --ttt_weight_dir ttt_augextra_0227_arlongcache_ntokens4 \
#     --ttt_weight_epoch 6 \
#     --select_tasks_path task_info_selected.csv

# # eval 0227_arlongcache_ntokens4 augextra epoch8
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
#     --tag epoch8 \
#     --weight_dir 0227_arlongcache_ntokens4 \
#     --weight_epoch 28 \
#     --ttt_weight_dir ttt_augextra_0227_arlongcache_ntokens4 \
#     --ttt_weight_epoch 8 \
#     --select_tasks_path task_info_selected.csv

# # eval 0227_arlongcache_ntokens4 augextra epoch10
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
#     --tag epoch10 \
#     --weight_dir 0227_arlongcache_ntokens4 \
#     --weight_epoch 28 \
#     --ttt_weight_dir ttt_augextra_0227_arlongcache_ntokens4 \
#     --ttt_weight_epoch 10 \
#     --select_tasks_path task_info_selected.csv




# EVALUATE BEST TTT WITH VOTING (can try different config for leave_ns)
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
#     --tag epoch5 \
#     --weight_dir 0227_arlongcache_ntokens4 \
#     --weight_epoch 28 \
#     --ttt_weight_dir ttt_augextra_0227_arlongcache_ntokens4 \
#     --ttt_weight_epoch 5 \
#     --select_tasks_path task_info_selected.csv
#     --select_tasks_path task_info_selected.csv
#     --leave_ns 1 \
#     --permute_n 2 \
#     --augment_n 5 \
#     --leave_ns_inc



# ttt
# Submitted batch job 58139583
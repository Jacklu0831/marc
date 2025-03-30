# for gs0, 1, 5, 25
# python make_sbatch.py --ngpu 1 --time 3 --bash_files bash_commands/0317_noprogram/0322_2_eval_ttt_gs_d8.sh

# for more
# python make_sbatch.py --ngpu 1 --time 7 --bash_files bash_commands/0317_noprogram/0322_2_eval_ttt_gs_d8.sh


# # gs1lr1e-3 0317_noprogram_base augd8epoch10
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --no_bos \
#     --tag augd8epoch10_gs1lr1e-3 \
#     --weight_dir 0317_noprogram_base \
#     --weight_epoch 24 \
#     --ttt_weight_dir ttt_augd8_0317_noprogram_base \
#     --ttt_weight_epoch 10 \
#     --select_tasks_path task_info_selected.csv \
#     --batch_size 16 \
#     --flash_attn \
#     --gs_batch_size 100000 \
#     --gs_iters 1 \
#     --gs_take_best \
#     --gs_lr 1e-3

# # gs5lr1e-3 0317_noprogram_base augd8epoch10
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --no_bos \
#     --tag augd8epoch10_gs5lr1e-3 \
#     --weight_dir 0317_noprogram_base \
#     --weight_epoch 24 \
#     --ttt_weight_dir ttt_augd8_0317_noprogram_base \
#     --ttt_weight_epoch 10 \
#     --select_tasks_path task_info_selected.csv \
#     --batch_size 16 \
#     --flash_attn \
#     --gs_batch_size 100000 \
#     --gs_iters 5 \
#     --gs_take_best \
#     --gs_lr 1e-3

# # gs25lr1e-3 0317_noprogram_base augd8epoch10
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
#     --no_bos \
#     --tag augd8epoch10_gs25lr1e-3 \
#     --weight_dir 0317_noprogram_base \
#     --weight_epoch 24 \
#     --ttt_weight_dir ttt_augd8_0317_noprogram_base \
#     --ttt_weight_epoch 10 \
#     --select_tasks_path task_info_selected.csv \
#     --batch_size 16 \
#     --flash_attn \
#     --gs_batch_size 100000 \
#     --gs_iters 25 \
#     --gs_take_best \
#     --gs_lr 1e-3

# gs100lr1e-3 0317_noprogram_base augd8epoch10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag augd8epoch10_gs100lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_noprogram_base \
    --ttt_weight_epoch 10 \
    --select_tasks_path task_info_selected.csv \
    --batch_size 16 \
    --flash_attn \
    --gs_batch_size 100000 \
    --gs_iters 100 \
    --gs_take_best \
    --gs_lr 1e-3

# gs250lr1e-3 0317_noprogram_base augd8epoch10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag augd8epoch10_gs250lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_noprogram_base \
    --ttt_weight_epoch 10 \
    --select_tasks_path task_info_selected.csv \
    --batch_size 16 \
    --flash_attn \
    --gs_batch_size 100000 \
    --gs_iters 250 \
    --gs_take_best \
    --gs_lr 1e-3


# Submitted batch job 58791787
# Submitted batch job 58791788
# Submitted batch job 58791789
# Submitted batch job 58791790
# Submitted batch job 58791791
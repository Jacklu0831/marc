# python make_sbatch.py --ngpu 1 --time 3 --bash_files bash_cmds/0401_arc/small/0419_1_ttt.sh



# ttt w high LR, save ckpts
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_savettt/test_time_evaluate.py \
    --select_tasks_path task_info_selected_easy.csv \
    --no_bos \
    --tag arc_test_save_ttt \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 5 \
    --ttt_lr 1e-2 \
    --ttt_permute_n 20 \
    --ttt_save

# {   'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.0,
#     'eval/exact_acc': 0.0,
#     'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/init_kv_time': 0.015376222133636475,
#     'eval/relaxed_token_acc': 0.15330239632463732,
#     'eval/token_acc': 0.0,
#     'eval/ttt_num_data': 7.8,
#     'eval/ttt_num_params': 84934656.0,
#     'eval/ttt_time': 1.951751720905304,
#     'eval/valid_grid': 0.0}

# load ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_savettt/test_time_evaluate.py \
    --select_tasks_path task_info_selected_easy.csv \
    --no_bos \
    --tag arc_test_save_ttt \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir eval_arc_test_save_ttt_0317_noprogram_base

# {   'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.0,
#     'eval/exact_acc': 0.0,
#     'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/init_kv_time': 0.03218364715576172,
#     'eval/relaxed_token_acc': 0.15330239632463732, <- good!
#     'eval/token_acc': 0.0,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0,
#     'eval/valid_grid': 0.0}
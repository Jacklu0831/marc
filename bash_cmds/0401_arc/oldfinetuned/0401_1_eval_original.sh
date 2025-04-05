# run locally

# arc gs0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24

# {   'eval/competition_all_acc': 0.1875,
#     'eval/competition_sub_acc': 0.1927710843373494,
#     'eval/correct_grid_dim': 0.8780487804878049,
#     'eval/exact_acc': 0.1951219512195122,
#     'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/relaxed_token_acc': 0.8099360188834345,
#     'eval/token_acc': 0.7355449331098669,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0,
#     'eval/valid_grid': 0.9634146341463414}
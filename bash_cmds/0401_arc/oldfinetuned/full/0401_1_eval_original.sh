# run locally

# arc gs0 full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --flash_attn \
    --tag arc_gs0_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24

# {   'eval/competition_all_acc': 0.1325,
#     'eval/competition_sub_acc': 0.1431980906921241,
#     'eval/correct_grid_dim': 0.9036144578313253,
#     'eval/exact_acc': 0.14457831325301204,
#     'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/relaxed_token_acc': 0.7945469393543263,
#     'eval/token_acc': 0.7259886154715124,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0,
#     'eval/valid_grid': 0.9710843373493976}
# run locally

# arc gs0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
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
#     'eval/init_kv_time': 0.019287754849689764,
#     'eval/relaxed_token_acc': 0.8099360188834345,
#     'eval/token_acc': 0.7355449331098669,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0,
#     'eval/valid_grid': 0.9634146341463414}




# debug runs on easy
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag test \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 50 \
    --gs_ntokens 32 \
    --gs_lr 1e-2

# finetuned:
# gsiter25 lr1e-3: 0.45
# gsiter25 lr1e-3 leaveoneout: 0.2
# gsiter25 lr1e-2 leaveoneout ntokens32: 0.15
# gsiter25 lr1e-1 leaveoneout ntokens32: 0.15
# gsiter25 lr3e-1 leaveoneout ntokens32: 0.15

# gsiter50 lr1e-3          : 0.4 -> 0.25
# gsiter50 lr1    ntokens32: 0.1
# gsiter50 lr3e-1 ntokens32: 0.45 -> 0.2625
# gsiter50 lr1e-1 ntokens32: 0.4
# gsiter50 lr3e-2 ntokens32: 0.35
# gsiter50 lr1e-2 ntokens32: 0.25
# gsiter50 lr3e-3 ntokens32: 0.1


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected_easy.csv \
    --no_bos \
    --tag test \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 50 \
    --gs_num_layer 12 \
    --gs_lr 1e-3

# gsiter50 lr2e-3 nlayer16: 0.4

# gsiter50 lr1e-3 nlayer4: 0.2
# gsiter50 lr1e-2 nlayer4: 0.4
# gsiter50 lr1e-1 nlayer4: 0.25

# gsiter50 lr1e-3 nlayer8: 0.35
# gsiter50 lr1e-2 nlayer8: 0.45
# gsiter50 lr1e-1 nlayer8: 0.05

# gsiter50 lr1e-3 nlayer12: 0.4
# gsiter50 lr1e-2 nlayer12: 0.45
# gsiter50 lr1e-1 nlayer12: 0.0

# gsiter50 lr1e-2 nlayer8 full: 0.275
# gsiter50 lr1e-2 nlayer12 full: 0.25
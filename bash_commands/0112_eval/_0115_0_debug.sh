accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/evaluate.py \
    --tag test_bs1 \
    --weight_dir manual_copy_0113_aug0.5 \
    --weight_epoch 6 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --data_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --select_tasks_path task_info_selected_easy.csv \
    --compact_grids \
    --max_seq_len 5120 \
    --tie_models \
    --flash_attn \
    --batch_size 1

# {   'eval/ce_loss': -1.0,
#     'eval/competition_all_acc': 0.05,
#     'eval/competition_sub_acc': 0.05,
#     'eval/correct_grid_dim': 0.85,
#     'eval/exact_acc': 0.05,
#     'eval/token_acc': 0.5148464136124741,
#     'eval/ttt_provided': 0.0,
#     'eval/valid_grid': 1.0}

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/evaluate.py \
    --tag test \
    --weight_dir manual_copy_0113_aug0.5 \
    --weight_epoch 6 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --data_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --select_tasks_path task_info_selected_easy.csv \
    --compact_grids \
    --max_seq_len 5120 \
    --tie_models \
    --flash_attn \
    --batch_size 2

# {   'eval/ce_loss': -1.0,
#     'eval/competition_all_acc': 0.05,
#     'eval/competition_sub_acc': 0.05,
#     'eval/correct_grid_dim': 0.8,
#     'eval/exact_acc': 0.05,
#     'eval/token_acc': 0.5148464136124741,
#     'eval/ttt_provided': 0.0,
#     'eval/valid_grid': 1.0}

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/evaluate.py \
    --tag test \
    --weight_dir manual_copy_0113_aug0.5 \
    --weight_epoch 6 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --data_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --select_tasks_path task_info_selected_easy.csv \
    --compact_grids \
    --max_seq_len 5120 \
    --tie_models \
    --flash_attn \
    --batch_size 2 \
    --leave_ns 0 \
    --permute_n 2 \
    --augment_n 5 \
    # --permute_iters 3

# exact same loss for multigpu and single gpu (permute_iters=0)
# {   'eval/ce_loss': -1.0,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.8387096774193549,
#     'eval/exact_acc': 0.012903225806451613,
#     'eval/token_acc': 0.5219202502105977,
#     'eval/ttt_provided': 0.0,
#     'eval/valid_grid': 0.9419354838709677}

# exact same loss for multigpu and single gpu (permute_iters=3)
# {   'eval/ce_loss': -1.0,
#     'eval/competition_all_acc': 0.0,
#     'eval/competition_sub_acc': 0.0,
#     'eval/correct_grid_dim': 0.8193548387096774,
#     'eval/exact_acc': 0.012903225806451613,
#     'eval/token_acc': 0.5123856558670051,
#     'eval/ttt_provided': 0.0,
#     'eval/valid_grid': 0.9354838709677419}

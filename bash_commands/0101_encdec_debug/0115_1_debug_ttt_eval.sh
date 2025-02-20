# eval test1 partiallora singlegpu
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag test_ttt1_partiallora \
    --weight_dir test_ttt_1 \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test1_partiallora_single_gpu_test_ttt_1 \
    --ttt_weight_epoch 100 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt_full \
    --decoder_ce_loss \
    --batch_size 2
# passed

# eval test1 partiallora multigpu
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag test_ttt1_partiallora \
    --weight_dir test_ttt_1 \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test1_partiallora_multi_gpu_test_ttt_1 \
    --ttt_weight_epoch 100 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt_full \
    --decoder_ce_loss \
    --batch_size 2
# passed

# eval test2 partiallora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/evaluate.py \
    --tag test_ttt2_partiallora \
    --weight_dir test_ttt_2 \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test2_partiallora_test_ttt_2 \
    --ttt_weight_epoch 100 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4_ttt/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method prefix2prefix \
    --decoder_ce_loss \
    --batch_size 1

# eval test1 fulllora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/evaluate.py \
    --tag test_ttt1_fulllora \
    --weight_dir test_ttt_1 \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test1_fulllora_test_ttt_1 \
    --ttt_weight_epoch 100 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4_ttt/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt_full \
    --decoder_ce_loss \
    --batch_size 1

# eval test2 fulllora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/evaluate.py \
    --tag test_ttt2_fulllora \
    --weight_dir test_ttt_2 \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test2_fulllora_test_ttt_2 \
    --ttt_weight_epoch 100 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4_ttt/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --conditioning_method prefix2prefix \
    --decoder_ce_loss \
    --batch_size 1

# test with more
accelerate launch --main_process_port $MASTER_PORT --num_processes 1 --mixed_precision bf16 encoder_decoder_new/evaluate.py \
    --tag test_ttt1_partiallora \
    --weight_dir test_ttt_1 \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test1_partiallora_test_ttt_1 \
    --ttt_weight_epoch 100 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4_ttt/training \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt_full \
    --decoder_ce_loss \
    --leave_ns 0 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc \
    --batch_size 1

# batch size 1
# {   'eval/ce_loss': 0.8805510012268011,
#     'eval/competition_all_acc': 1.0,
#     'eval/competition_sub_acc': 1.0,
#     'eval/correct_grid_dim': 0.5416666666666666,
#     'eval/exact_acc': 0.25,
#     'eval/token_acc': 0.4351851851851853,
#     'eval/ttt_provided': 1.0,
#     'eval/valid_grid': 0.5416666666666666}
# python make_sbatch.py --ngpu 1 --time 12 --bash_files bash_commands/0112_encdec_eval/0117_1_ntoken1_gs_batchsize1.sh

# 400task gs1 beta0.9 batchsize1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_gs1_batchsize1_beta0.9 \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --data_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --batch_size 1 \
    --gs_iters 1 \
    --gs_beta2 0.9 \
    --gs_batch_size 1

# 400task gs5 beta0.9 batchsize1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_gs5_batchsize1_beta0.9 \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --data_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --batch_size 1 \
    --gs_iters 5 \
    --gs_beta2 0.9 \
    --gs_batch_size 1

# 400task gs25 beta0.9 batchsize1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_gs25_batchsize1_beta0.9 \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --data_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --batch_size 1 \
    --gs_iters 25 \
    --gs_beta2 0.9 \
    --gs_batch_size 1

# 400task gs100 beta0.9 batchsize1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_gs100_batchsize1_beta0.9 \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
<<<<<<< HEAD
    --data_dir /scratch/zy3101/re-arc/arc_original/evaluation \
=======
    --data_dir ./data/re-arc/arc_original/evaluation \
>>>>>>> origin/main
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --batch_size 1 \
    --gs_iters 100 \
    --gs_beta2 0.9 \
    --gs_batch_size 1

# Submitted batch job 55989682 # 0.0075
# Submitted batch job 55989683 # 0.0095
# Submitted batch job 55989684 # 0.0125
# Submitted batch job 55989685 # 0.0175
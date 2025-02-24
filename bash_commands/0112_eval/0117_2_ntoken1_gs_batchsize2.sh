# python make_sbatch.py --ngpu 1 --time 12 --bash_files bash_commands/0112_encdec_eval/0117_2_ntoken1_gs_batchsize2.sh

# 400task gs1 beta0.9 batchsize2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_gs1_batchsize2_beta0.9 \
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
    --gs_batch_size 2

# 400task gs5 beta0.9 batchsize2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_gs5_batchsize2_beta0.9 \
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
    --gs_batch_size 2

# 400task gs25 beta0.9 batchsize2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_gs25_batchsize2_beta0.9 \
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
    --gs_batch_size 2

# 400task gs100 beta0.9 batchsize2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_gs100_batchsize2_beta0.9 \
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
    --gs_batch_size 2

# Submitted batch job 55989768 # 0.005
# Submitted batch job 56008179
# Submitted batch job 56008180
# Submitted batch job 55989771 # 0.0175
# python make_sbatch.py --ngpu 1 --time 1 --single --bash_files bash_cmds/0401_arc/small/0401_10_gs_numpermute_concat.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# arc numpermute2 concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_numpermute2_concat \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --num_permute 2 \
    --permute_batch_size 2

# arc numpermute4 concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_numpermute4_concat \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --num_permute 4 \
    --permute_batch_size 4

# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_arc/small/0420_0_evalondemon.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# arc evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_evalondemon \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --eval_on_demonstrations

# arc ttt iter250 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter250_evalondemon \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 250 \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# 0.1375
# Submitted batch job 59604210
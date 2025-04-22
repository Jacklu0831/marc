# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_mmlu/pretrained/0419_1_ttt.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)









# mmlu ttt iter5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter5 \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# mmlu ttt iter10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# mmlu ttt iter15
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter15 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# mmlu ttt iter20
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# mmlu ttt iter25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter25 \
    --ttt_iters 25 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# 5: 42.984005275866096
# 10: 44.611232390075415
# 15: 44.61005736526445
# 20: 43.42785560347695
# 25: 42.54093571913059